use std::collections::HashMap;

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
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::Compiler;

struct FunctionLocalInfo<'ctx> {
    alloc: PointerValue<'ctx>,
    defining_block: BasicBlock<'ctx>,
}

impl<'ctx> FunctionLocalInfo<'ctx> {
    fn new(alloc: PointerValue<'ctx>, defining_block: BasicBlock<'ctx>) -> Self {
        Self {
            alloc,
            defining_block,
        }
    }
}

pub struct LLVM<'ctx> {
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    function_locals: HashMap<mir::LocalId, FunctionLocalInfo<'ctx>>,
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

                let i32_type = self.ctx().i32_type();
                self.builder
                    .build_load(i32_type, local.alloc, "load")
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

                let alloca = self
                    .builder
                    .build_alloca(value.get_type(), format!("x{}", localid).as_str())
                    .unwrap();

                let block = llvm_fn.get_last_basic_block().unwrap();
                let local_info = FunctionLocalInfo::new(alloca, block);
                self.function_locals.insert(*localid, local_info);
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

                let phi_alloca = self.builder.build_alloca(i32_type, "alloca").unwrap();
                self.builder
                    .build_store(phi_alloca, phi_value.as_basic_value())
                    .unwrap();

                let block = llvm_fn.get_last_basic_block().unwrap();
                let local_info = FunctionLocalInfo::new(phi_alloca, block);
                self.function_locals.insert(*localid, local_info);
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
            _ => todo!(),
        }

        Ok(())
    }

    fn compile_block(
        &mut self,
        block: mir::BasicBlock,
        llvm_fn: &FunctionValue<'ctx>,
        ctx: &FrontendCtx,
    ) -> Result<()> {
        self.ctx()
            .append_basic_block(*llvm_fn, &format!("bb {}", block.block_id));

        for stmt in block.stmts.iter() {
            self.compile_statement(stmt, llvm_fn, ctx)?;
        }

        self.compile_terminator(block.terminator, llvm_fn, ctx)?;

        Ok(())
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
        let llvm_fn = self.module.add_function(
            format!("{}", function.function_id).as_str(),
            fn_type,
            Some(Linkage::External),
        );

        for block in function.into_iter() {
            self.compile_block(block, &llvm_fn, ctx)?;
        }

        llvm_fn.print_to_stderr();

        Ok(())
    }
}

impl<'ctx> Compiler for LLVM<'ctx> {
    fn compile(mut self, module: mir::Module, ctx: FrontendCtx) -> Result<()> {
        for function in module.functions {
            self.compile_function(function, &ctx)?;
        }

        self.module.verify().unwrap();
        Ok(())
    }
}
pub fn compile(module: mir::Module, ctx: FrontendCtx) -> Result<()> {
    let context = Context::create();

    let llvm = LLVM::new(&context);
    llvm.compile(module, ctx)?;

    Ok(())
}
