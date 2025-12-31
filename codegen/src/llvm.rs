use std::collections::HashMap;
use std::path::Path;

use frontend::mir::AllocKind;
use frontend::{BinOp, Ctx as FrontendCtx, mir};

use anyhow::Result;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::{Context, ContextRef};
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::targets::{
    ByteOrdering, CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{
    AnyType, AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType,
};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::Compiler;

struct FunctionLocalInfo<'ctx> {
    ty: BasicTypeEnum<'ctx>,
    alloc: PointerValue<'ctx>,
    defining_block: BasicBlock<'ctx>,
}

impl<'ctx> FunctionLocalInfo<'ctx> {
    fn new(
        ty: BasicTypeEnum<'ctx>,
        alloc: PointerValue<'ctx>,
        defining_block: BasicBlock<'ctx>,
    ) -> Self {
        Self {
            ty,
            alloc,
            defining_block,
        }
    }
}

pub struct LLVM<'ctx> {
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    function_locals: HashMap<mir::LocalId, FunctionLocalInfo<'ctx>>,
    function_blocks: Vec<BasicBlock<'ctx>>,
    target_machine: TargetMachine,
}

impl<'ctx> LLVM<'ctx> {
    fn new(ctx: &'ctx Context) -> Self {
        let module = ctx.create_module("mnemo");
        let builder = ctx.create_builder();

        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).unwrap();

        let target_machine = target
            .create_target_machine(
                &triple,
                "generic",
                "",
                OptimizationLevel::None,
                RelocMode::Default,
                CodeModel::Default,
            )
            .expect("to create a target machine");

        Self {
            module,
            builder,
            target_machine,
            function_locals: HashMap::new(),
            function_blocks: Vec::new(),
        }
    }

    fn ctx(&self) -> ContextRef<'ctx> {
        self.module.get_context()
    }

    fn basic_type_to_llvm_basic_type(ctx: ContextRef<'ctx>, ty: &mir::Ty) -> BasicTypeEnum<'ctx> {
        match ty {
            mir::Ty::Int => ctx.i32_type().as_basic_type_enum(),
            mir::Ty::Bool => ctx.bool_type().as_basic_type_enum(),
            mir::Ty::Char => ctx.i8_type().as_basic_type_enum(),
            mir::Ty::Unit => panic!("Unit type not supported as a basic type"),
            mir::Ty::Array(ty, len) => {
                let ty = Self::basic_type_to_llvm_basic_type(ctx, ty);
                ty.array_type(*len as u32).as_basic_type_enum()
            }
            mir::Ty::Ptr(_) => ctx.ptr_type(AddressSpace::default()).as_basic_type_enum(),
            _ => todo!(),
        }
    }

    fn basic_type_to_llvm_basic_metadata_type(
        ctx: ContextRef<'ctx>,
        ty: &mir::Ty,
    ) -> BasicMetadataTypeEnum<'ctx> {
        Self::basic_type_to_llvm_basic_type(ctx, ty).into()
    }

    fn type_to_fn_type(
        ctx: ContextRef<'ctx>,
        ty: &mir::Ty,
        param_types: Vec<BasicMetadataTypeEnum<'ctx>>,
    ) -> FunctionType<'ctx> {
        match ty {
            mir::Ty::Int => ctx.i32_type().fn_type(&param_types, false),
            mir::Ty::Bool => ctx.bool_type().fn_type(&param_types, false),
            mir::Ty::Char => ctx.i8_type().fn_type(&param_types, false),
            mir::Ty::Unit => ctx.void_type().fn_type(&param_types, false),
            mir::Ty::Ptr(_) => ctx
                .ptr_type(AddressSpace::default())
                .fn_type(&param_types, false),

            _ => todo!(),
        }
    }

    fn compile_operand(&self, operand: &mir::Operand) -> Result<BasicValueEnum<'ctx>> {
        let value = match operand {
            mir::Operand::Constant(c) => self
                .ctx()
                .i32_type()
                .const_int(*c as u64, false)
                .as_basic_value_enum(),
            mir::Operand::Local(local_id) => {
                let local = self.function_locals.get(local_id).unwrap();

                self.builder
                    .build_load(local.ty, local.alloc, "load")
                    .unwrap()
                    .as_basic_value_enum()
            }
        };

        Ok(value)
    }

    fn compile_rvalue(
        &self,
        rvalue: &mir::RValue,
        llvm_fn: &FunctionValue<'ctx>,
        ctx: &FrontendCtx,
    ) -> Result<BasicValueEnum<'ctx>> {
        let int_val = match rvalue {
            mir::RValue::Use(operand) => self.compile_operand(operand)?,
            mir::RValue::BinOp(binop, lhs, rhs) => {
                let lhs = self.compile_operand(lhs)?;
                let rhs = self.compile_operand(rhs)?;

                match binop {
                    BinOp::Add => self
                        .builder
                        // TOOD: we are assuming ints here
                        .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add")
                        .unwrap()
                        .as_basic_value_enum(),
                    _ => todo!(),
                }
            }
            mir::RValue::Alloc(alloc_kind, operands) => {
                match alloc_kind {
                    AllocKind::Array(ty_hint) => {
                        let elem_type = Self::basic_type_to_llvm_basic_type(self.ctx(), ty_hint);

                        let array_len = operands.len() as u32;
                        let array_type = elem_type.array_type(array_len);

                        // TODO: for mem2reg, allocas should ideally happen
                        // in the function's entry block, not the current builder position.
                        let array_ptr = self.builder.build_alloca(array_type, "array_alloca")?;

                        let i32_type = self.ctx().i32_type();
                        let zero = i32_type.const_zero();

                        for (i, operand) in operands.iter().enumerate() {
                            let val = self.compile_operand(operand)?;

                            let index = i32_type.const_int(i as u64, false);

                            let elem_ptr = unsafe {
                                self.builder.build_gep(
                                    array_type,
                                    array_ptr,
                                    &[zero, index],
                                    "elem_ptr",
                                )?
                            };

                            self.builder.build_store(elem_ptr, val)?;
                        }

                        array_ptr.as_basic_value_enum()
                    }
                    _ => todo!(),
                }
            }
        };

        Ok(int_val)
    }

    fn compile_statement(
        &mut self,
        stmt: &mir::Statement,
        llvm_fn: &FunctionValue<'ctx>,
        ctx: &FrontendCtx,
    ) -> Result<()> {
        self.builder
            .position_at_end(llvm_fn.get_last_basic_block().unwrap());
        match stmt {
            mir::Statement::Assign(localid, rvalue) => {
                let value = self.compile_rvalue(rvalue, llvm_fn, ctx)?;

                let alloca = self.function_locals.get(localid).unwrap().alloc;

                self.builder.build_store(alloca, value).unwrap();
            }
            mir::Statement::Phi(localid, local_ids) => {
                let i32_type = self.ctx().i32_type();
                let phi_value = self
                    .builder
                    .build_phi(i32_type, format!("phi_{}", localid).as_str())
                    .unwrap();

                for local_id in local_ids {
                    let local = self.function_locals.get(local_id).unwrap();
                    phi_value.add_incoming(&[(&local.alloc, local.defining_block)]);
                }

                let alloca = self.function_locals.get(localid).unwrap().alloc;
                self.builder
                    .build_store(alloca, phi_value.as_basic_value())
                    .unwrap();
            }
        }

        Ok(())
    }

    fn compile_terminator(
        &mut self,
        terminator: mir::Terminator,
        llvm_fn: &FunctionValue<'ctx>,
        ctx: &FrontendCtx,
    ) -> Result<()> {
        match terminator {
            mir::Terminator::Return => {
                let _ = self.builder.build_return(None);
            }
            mir::Terminator::Call {
                function_name,
                args,
                destination,
                target,
            } => {
                let fn_val = self.module.get_function(function_name.as_str()).unwrap();

                let mut call_args = Vec::new();
                for rval in args {
                    let basic_elem_type = self.compile_rvalue(&rval, llvm_fn, ctx)?;
                    call_args.push(basic_elem_type.into());
                }

                let call = self.builder.build_call(fn_val, &call_args, "call_result")?;

                if let Some(destination) = destination {
                    let alloc = self.function_locals.get(&destination).unwrap().alloc;
                    //
                    // SAFETY: destination should only be SOME if the call returns
                    let call_value = call.try_as_basic_value().basic().unwrap();
                    self.builder.build_store(alloc, call_value)?;
                }

                let target_block = self
                    .ctx()
                    .append_basic_block(*llvm_fn, &format!("bb{}", target));
                self.function_blocks.push(target_block);

                self.builder.build_unconditional_branch(target_block)?;
            }
            _ => todo!(),
        }

        Ok(())
    }

    fn compile_block(
        &mut self,
        block: mir::BasicBlock,
        llvm_fn: &FunctionValue<'ctx>,
        ctx: &FrontendCtx,
    ) -> Result<BasicBlock<'ctx>> {
        println!(
            "compiling block {} with len {}",
            block.block_id,
            self.function_blocks.len()
        );
        let bb = if self.function_blocks.len() == block.block_id {
            self.function_blocks[block.block_id - 1]
        } else {
            let bb = self
                .ctx()
                .append_basic_block(*llvm_fn, &format!("bb{}", block.block_id));
            self.function_blocks.push(bb);

            bb
        };
        self.builder.position_at_end(bb);

        for stmt in block.stmts.iter() {
            self.compile_statement(stmt, llvm_fn, ctx)?;
        }

        self.compile_terminator(block.terminator, llvm_fn, ctx)?;

        Ok(bb)
    }

    fn compile_function(&mut self, function: mir::Function, ctx: &FrontendCtx) -> Result<()> {
        self.function_locals.clear();

        let mut arg_types = Vec::new();
        let llvm_ctx = self.ctx();

        for (_i, local) in function.locals.iter().enumerate().take(function.parameters) {
            arg_types.push(Self::basic_type_to_llvm_basic_metadata_type(
                llvm_ctx, &local.ty,
            ));
        }

        let fn_type = Self::type_to_fn_type(llvm_ctx, &function.return_ty, arg_types);
        let llvm_fn =
            self.module
                .add_function(function.name.as_str(), fn_type, Some(Linkage::External));

        let entry_block = self.ctx().append_basic_block(llvm_fn, "entry");
        self.builder.position_at_end(entry_block);

        for local in function.locals.iter().skip(function.parameters) {
            let local_ty = Self::basic_type_to_llvm_basic_type(llvm_ctx, &local.ty);
            let local_alloca = self
                .builder
                .build_alloca(local_ty, format!("x{}", local.id).as_str())?;

            let local_info = FunctionLocalInfo::new(local_ty, local_alloca, entry_block);
            self.function_locals.insert(local.id, local_info);
        }

        for block in function.into_iter() {
            let bb = self.compile_block(block, &llvm_fn, ctx)?;
            if entry_block.get_terminator().is_none() {
                self.builder.position_at_end(entry_block);
                self.builder.build_unconditional_branch(bb).unwrap();
                self.builder.position_at_end(bb);
            }
        }

        Ok(())
    }

    fn compile_extern(&mut self, extern_: mir::Extern) -> Result<()> {
        let param_types = extern_
            .params
            .iter()
            .map(|ty| Self::basic_type_to_llvm_basic_metadata_type(self.ctx(), ty))
            .collect();

        let fn_ty = Self::type_to_fn_type(self.ctx(), &extern_.return_ty, param_types);
        let _ = self.module.add_function(&extern_.name, fn_ty, None);
        Ok(())
    }

    fn output(self) {
        let triple = TargetMachine::get_default_triple();

        self.module
            .set_data_layout(&self.target_machine.get_target_data().get_data_layout());
        self.module.set_triple(&triple);

        let path = Path::new("a.o");
        let _ = self
            .target_machine
            .write_to_file(&self.module, FileType::Object, path);
    }
}

impl<'ctx> Compiler for LLVM<'ctx> {
    fn compile(mut self, module: mir::Module, ctx: FrontendCtx) -> Result<()> {
        for extern_ in module.externs {
            self.compile_extern(extern_)?;
        }

        for function in module.functions {
            self.compile_function(function, &ctx)?;
        }

        self.module.print_to_stderr();
        self.module.verify().unwrap();

        self.output();

        Ok(())
    }
}
pub fn compile(module: mir::Module, ctx: FrontendCtx) -> Result<()> {
    let context = Context::create();

    let llvm = LLVM::new(&context);
    llvm.compile(module, ctx)?;

    Ok(())
}
