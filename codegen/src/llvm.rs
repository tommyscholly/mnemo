use std::collections::HashMap;
use std::path::Path;

use frontend::mir::AllocKind;
use frontend::{BinOp, Ctx as FrontendCtx, mir};

use anyhow::Result;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::{Context, ContextRef};
use inkwell::module::{Linkage, Module};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::Compiler;

struct FunctionLocalInfo<'ctx> {
    ty: BasicTypeEnum<'ctx>,
    tk: mir::Ty,
    alloc: PointerValue<'ctx>,
    defining_block: BasicBlock<'ctx>,
}

impl<'ctx> FunctionLocalInfo<'ctx> {
    fn new(
        ty: BasicTypeEnum<'ctx>,
        tk: mir::Ty,
        alloc: PointerValue<'ctx>,
        defining_block: BasicBlock<'ctx>,
    ) -> Self {
        Self {
            ty,
            tk,
            alloc,
            defining_block,
        }
    }
}

pub struct Llvm<'ctx> {
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    function_locals: HashMap<mir::LocalId, FunctionLocalInfo<'ctx>>,
    function_blocks: HashMap<mir::BlockId, BasicBlock<'ctx>>,
    target_machine: TargetMachine,
}

impl<'ctx> Llvm<'ctx> {
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
            function_blocks: HashMap::new(),
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
            mir::Ty::Record(tys) => {
                let field_tys: Vec<BasicTypeEnum<'ctx>> = tys
                    .iter()
                    .map(|t| Self::basic_type_to_llvm_basic_type(ctx, t))
                    .collect();

                ctx.struct_type(&field_tys, false).as_basic_type_enum()
            }
            mir::Ty::TaggedUnion(_) => {
                let union_size = ty.bytes();

                let tag_ty = ctx.i8_type().as_basic_type_enum();
                let payload_ty = ctx
                    .i8_type()
                    .array_type(union_size as u32)
                    .as_basic_type_enum();

                ctx.struct_type(&[tag_ty, payload_ty], false)
                    .as_basic_type_enum()
            }
            mir::Ty::Tuple(tys) => {
                let field_tys: Vec<BasicTypeEnum<'ctx>> = tys
                    .iter()
                    .map(|t| Self::basic_type_to_llvm_basic_type(ctx, t))
                    .collect();

                ctx.struct_type(&field_tys, false).as_basic_type_enum()
            }
            mir::Ty::Str => ctx.ptr_type(AddressSpace::default()).as_basic_type_enum(),
            mir::Ty::Variadic => unreachable!(),
            ty => panic!("unimplemented type {:?}", ty),
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
        is_variadic: bool,
    ) -> FunctionType<'ctx> {
        match ty {
            mir::Ty::Int => ctx.i32_type().fn_type(&param_types, is_variadic),
            mir::Ty::Bool => ctx.bool_type().fn_type(&param_types, is_variadic),
            mir::Ty::Char => ctx.i8_type().fn_type(&param_types, is_variadic),
            mir::Ty::Unit => ctx.void_type().fn_type(&param_types, is_variadic),
            mir::Ty::Ptr(_) => ctx
                .ptr_type(AddressSpace::default())
                .fn_type(&param_types, is_variadic),

            mir::Ty::Array(ty, len) => {
                let ty = Self::basic_type_to_llvm_basic_type(ctx, ty);
                ty.array_type(*len as u32)
                    .fn_type(&param_types, is_variadic)
            }
            _ => todo!(),
        }
    }

    fn get_or_create_bb(
        &mut self,
        llvm_fn: &FunctionValue<'ctx>,
        block_id: mir::BlockId,
    ) -> BasicBlock<'ctx> {
        if let Some(bb) = self.function_blocks.get(&block_id) {
            return *bb;
        }

        let bb = self
            .ctx()
            .append_basic_block(*llvm_fn, format!("bb{}", block_id).as_str());

        self.function_blocks.insert(block_id, bb);
        bb
    }

    fn load_place(&self, place: &mir::Place) -> Result<BasicValueEnum<'ctx>> {
        let local = self.function_locals.get(&place.local).unwrap();
        let load = match &place.kind {
            mir::PlaceKind::Deref => self
                .builder
                .build_load(local.ty, local.alloc, "load")
                .unwrap()
                .as_basic_value_enum(),
            mir::PlaceKind::Field(idx, ty) => {
                let field_alloca = self.builder.build_struct_gep(
                    local.ty,
                    local.alloc,
                    *idx as u32,
                    "field_alloca",
                )?;

                let load_ty = Self::basic_type_to_llvm_basic_type(self.ctx(), ty);

                self.builder
                    .build_load(
                        load_ty,
                        field_alloca,
                        format!("load_field_{}", idx).as_str(),
                    )
                    .unwrap()
                    .as_basic_value_enum()
            }
            mir::PlaceKind::Index(local_idx) => {
                let pointee_ty = local.ty.into_array_type().get_element_type();

                let index_local = self.function_locals.get(local_idx).unwrap();
                let index_val = self
                    .builder
                    .build_load(index_local.ty, index_local.alloc, "index_load")?
                    .into_int_value();
                let zero = self.ctx().i32_type().const_zero();

                let gep = unsafe {
                    self.builder.build_in_bounds_gep(
                        local.ty,
                        local.alloc,
                        &[zero, index_val],
                        "arr_index_gep",
                    )
                }?;

                self.builder
                    .build_load(pointee_ty, gep, "arr_index_load")?
                    .as_basic_value_enum()
            }
        };

        Ok(load)
    }

    fn compile_operand(&self, operand: &mir::Operand) -> Result<BasicValueEnum<'ctx>> {
        let value = match operand {
            mir::Operand::Constant(constant) => match constant {
                mir::Constant::Int(i) => self
                    .ctx()
                    .i32_type()
                    .const_int(*i as u64, false)
                    .as_basic_value_enum(),
                mir::Constant::Bool(b) => self
                    .ctx()
                    .bool_type()
                    .const_int(*b as u64, false)
                    .as_basic_value_enum(),
            },

            // copies are trivial loads
            mir::Operand::Copy(place) => self.load_place(place)?,
        };

        Ok(value)
    }

    fn compile_rvalue(
        &self,
        rvalue: &mir::RValue,
        _destination: Option<PointerValue<'ctx>>,
        _llvm_fn: &FunctionValue<'ctx>,
        _ctx: &FrontendCtx,
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
                                self.builder.build_in_bounds_gep(
                                    array_type,
                                    array_ptr,
                                    &[zero, index],
                                    "elem_ptr",
                                )?
                            };

                            self.builder.build_store(elem_ptr, val)?;
                        }

                        // match destination {
                        //     Some(destination) => {
                        //         self.builder.build_store(destination, array_ptr)?;
                        //         destination.as_basic_value_enum()
                        //     }
                        //     // if there is no destination, we just return the array ptr, as in calls
                        //     None => array_ptr.as_basic_value_enum(),
                        // }

                        // copy array into memory to be assigned in compile_statement, this is inefficient
                        self.builder
                            .build_load(array_type, array_ptr, "array_load")?
                            .as_basic_value_enum()
                    }
                    AllocKind::Record(fields) => {
                        let field_tys: Vec<BasicTypeEnum<'ctx>> = fields
                            .iter()
                            .map(|ty| Self::basic_type_to_llvm_basic_type(self.ctx(), ty))
                            .collect();

                        let struct_ty = self.ctx().struct_type(&field_tys, false);

                        let struct_ptr = self.builder.build_alloca(struct_ty, "struct_alloca")?;

                        for (i, operand) in operands.iter().take(fields.len()).enumerate() {
                            let field_alloca = self.builder.build_struct_gep(
                                struct_ty,
                                struct_ptr,
                                i as u32,
                                "field_alloca",
                            )?;
                            self.builder
                                .build_store(field_alloca, self.compile_operand(operand)?)?;
                        }

                        self.builder
                            .build_load(struct_ty, struct_ptr, "struct_load")?
                    }
                    AllocKind::Variant(tag, payload_ty) => {
                        let tag_ty = self.ctx().i8_type().as_basic_type_enum();
                        let variant_ty = match payload_ty {
                            mir::Ty::Unit => self.ctx().struct_type(&[tag_ty], false),
                            _ => {
                                // Calculate payload size accounting for alignment.
                                // The payload type determines both size and alignment requirements.
                                // For proper union semantics, we use the payload's alignment
                                // to determine padding, ensuring consistent layout with local variables.
                                let payload_size = payload_ty.bytes();
                                let payload_align = payload_ty.align();

                                // Calculate total struct size: 1 byte tag + padding + payload
                                // This must match what Ty::TaggedUnion::bytes() returns
                                let base_size = 1 + payload_size;
                                let padding = if payload_align > 1 {
                                    (payload_align - (base_size % payload_align)) % payload_align
                                } else {
                                    0
                                };
                                let total_size = base_size + padding;

                                // The array size should match what basic_type_to_llvm_basic_type
                                // uses for TaggedUnion, which is ty.bytes() directly
                                let array_size = total_size;

                                let payload_ty = self
                                    .ctx()
                                    .i8_type()
                                    .array_type(array_size as u32)
                                    .as_basic_type_enum();
                                self.ctx().struct_type(&[tag_ty, payload_ty], false)
                            }
                        };

                        let variant_ptr =
                            self.builder.build_alloca(variant_ty, "variant_alloca")?;

                        let tag_val = self.ctx().i8_type().const_int(*tag as u64, false);
                        self.builder.build_store(variant_ptr, tag_val)?;

                        if operands.len() > 1 {
                            todo!("payloads with more than one operand not implemented");
                        } else if operands.len() == 1 {
                            let payload_gep = self.builder.build_struct_gep(
                                variant_ty,
                                variant_ptr,
                                1,
                                "payload_gep",
                            )?;

                            let payload_val = self.compile_operand(&operands[0])?;
                            self.builder.build_store(payload_gep, payload_val)?;
                        }

                        self.builder
                            .build_load(variant_ty, variant_ptr, "variant_load")?
                    }
                    AllocKind::Tuple(tys) => {
                        let field_tys: Vec<BasicTypeEnum<'ctx>> = tys
                            .iter()
                            .map(|t| Self::basic_type_to_llvm_basic_type(self.ctx(), t))
                            .collect();

                        let struct_ty = self.ctx().struct_type(&field_tys, false);

                        let struct_ptr = self.builder.build_alloca(struct_ty, "struct_alloca")?;

                        for (i, operand) in operands.iter().take(tys.len()).enumerate() {
                            let field_alloca = self.builder.build_struct_gep(
                                struct_ty,
                                struct_ptr,
                                i as u32,
                                "field_alloca",
                            )?;
                            self.builder
                                .build_store(field_alloca, self.compile_operand(operand)?)?;
                        }

                        self.builder
                            .build_load(struct_ty, struct_ptr, "struct_load")?
                    }
                    AllocKind::Str(s) => self
                        .builder
                        .build_global_string_ptr(s, format!("str_{}", s).as_str())?
                        .as_pointer_value()
                        .as_basic_value_enum(),
                    a => panic!("unimplemented alloc kind {:?}", a),
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
        match stmt {
            mir::Statement::Assign(localid, rvalue) => {
                let alloca = self.function_locals.get(localid).unwrap().alloc;
                let value = self.compile_rvalue(rvalue, Some(alloca), llvm_fn, ctx)?;

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
            mir::Statement::Call {
                function_name,
                args,
                destination,
            } => {
                let fn_val = self.module.get_function(function_name.as_str()).unwrap();

                let mut call_args = Vec::new();
                for rval in args {
                    let basic_elem_type = self.compile_rvalue(rval, None, llvm_fn, ctx)?;
                    call_args.push(basic_elem_type.into());
                }

                let call = self.builder.build_call(fn_val, &call_args, "call_result")?;

                if let Some(destination) = destination {
                    let alloc = self.function_locals.get(destination).unwrap().alloc;
                    //
                    // SAFETY: destination should only be SOME if the call returns
                    let call_value = call.try_as_basic_value().basic().unwrap();
                    self.builder.build_store(alloc, call_value)?;
                }
            }
        }

        Ok(())
    }

    fn compile_terminator(
        &mut self,
        terminator: mir::Terminator,
        llvm_fn: &FunctionValue<'ctx>,
        _ctx: &FrontendCtx,
    ) -> Result<()> {
        match terminator {
            mir::Terminator::Return(local_id) => match local_id {
                Some(local_id) => {
                    println!("local_id: {:?}", local_id);
                    println!("fn name {:?}", llvm_fn.get_name());
                    let local = self.function_locals.get(&local_id).unwrap();
                    let val = self
                        .builder
                        .build_load(local.ty, local.alloc, "return_val")
                        .unwrap();
                    self.builder.build_return(Some(&val))?;
                }
                None => {
                    self.builder.build_return(None)?;
                }
            },
            mir::Terminator::BrIf(cond_local_id, then_block, else_block) => {
                let cond_local = self.function_locals.get(&cond_local_id).unwrap();
                let cond_val = self
                    .builder
                    .build_load(cond_local.ty, cond_local.alloc, "cond_val")
                    .unwrap();

                let then_bb = self.get_or_create_bb(llvm_fn, then_block);
                let else_bb = self.get_or_create_bb(llvm_fn, else_block);

                self.builder.build_conditional_branch(
                    cond_val.into_int_value(),
                    then_bb,
                    else_bb,
                )?;
            }
            mir::Terminator::Br(target_block) => {
                let target_block = self.get_or_create_bb(llvm_fn, target_block);
                self.builder.build_unconditional_branch(target_block)?;
            }
            mir::Terminator::BrTable(local_id, jump_table) => {
                let jump_local_info = self.function_locals.get(&local_id).unwrap();
                let (jump_val, switch_type) = match jump_local_info.tk {
                    mir::Ty::Int => {
                        let switch_type = self.ctx().i32_type();

                        let jump_val = self
                            .builder
                            .build_load(switch_type, jump_local_info.alloc, "jump_val")
                            .unwrap()
                            .into_int_value();

                        (jump_val, switch_type)
                    }
                    mir::Ty::TaggedUnion(_) => {
                        let switch_type = self.ctx().i8_type();

                        let jump_ptr = self.builder.build_struct_gep(
                            jump_local_info.ty,
                            jump_local_info.alloc,
                            0,
                            "jump_ptr",
                        )?;

                        let jump_val = self
                            .builder
                            .build_load(switch_type, jump_ptr, "jump_val")
                            .unwrap()
                            .into_int_value();

                        (jump_val, switch_type)
                    }
                    _ => todo!(),
                };

                let default_bb = self.get_or_create_bb(llvm_fn, jump_table.default);

                let mut cases = Vec::new();
                for (val, block_id) in jump_table.cases {
                    let bb = self.get_or_create_bb(llvm_fn, block_id);

                    let val = switch_type.const_int(val as u64, false);
                    cases.push((val, bb));
                }

                self.builder.build_switch(jump_val, default_bb, &cases)?;
            }
        }

        Ok(())
    }

    fn compile_block(
        &mut self,
        block: mir::BasicBlock,
        llvm_fn: &FunctionValue<'ctx>,
        ctx: &FrontendCtx,
    ) -> Result<BasicBlock<'ctx>> {
        let bb = self.get_or_create_bb(llvm_fn, block.block_id);

        self.builder.position_at_end(bb);

        for stmt in block.stmts.iter() {
            self.compile_statement(stmt, llvm_fn, ctx)?;
        }

        self.compile_terminator(block.terminator, llvm_fn, ctx)?;

        Ok(bb)
    }

    fn precompute_function(&mut self, function: &mir::Function) {
        let mut arg_types = Vec::new();
        let llvm_ctx = self.ctx();

        let mut is_variadic = false;
        for (_i, local) in function.locals.iter().enumerate().take(function.parameters) {
            match &local.ty {
                mir::Ty::Variadic => {
                    is_variadic = true;
                }
                _ => {
                    arg_types.push(Self::basic_type_to_llvm_basic_metadata_type(
                        llvm_ctx, &local.ty,
                    ));
                }
            }
        }

        let fn_type = Self::type_to_fn_type(llvm_ctx, &function.return_ty, arg_types, is_variadic);
        let _ = self
            .module
            .add_function(function.name.as_str(), fn_type, Some(Linkage::External));
    }

    fn compile_function(&mut self, function: mir::Function, ctx: &FrontendCtx) -> Result<()> {
        self.function_locals.clear();
        self.function_blocks.clear();

        let llvm_ctx = self.ctx();
        let llvm_fn = self.module.get_function(&function.name).unwrap();

        let entry_block = self
            .ctx()
            .append_basic_block(llvm_fn, format!("{}_entry", function.name).as_str());
        self.builder.position_at_end(entry_block);

        for (param_idx, param_local) in function.locals.iter().take(function.parameters).enumerate()
        {
            let local_ty = Self::basic_type_to_llvm_basic_type(llvm_ctx, &param_local.ty);
            let local_alloca = self
                .builder
                .build_alloca(local_ty, format!("param{}", param_local.id).as_str())?;

            let param = llvm_fn.get_nth_param(param_idx as u32).unwrap();

            self.builder.build_store(local_alloca, param).unwrap();

            let local_info =
                FunctionLocalInfo::new(local_ty, param_local.ty.clone(), local_alloca, entry_block);
            self.function_locals.insert(param_local.id, local_info);
        }

        for local in function.locals.iter().skip(function.parameters) {
            let local_ty = Self::basic_type_to_llvm_basic_type(llvm_ctx, &local.ty);
            let local_alloca = self
                .builder
                .build_alloca(local_ty, format!("x{}", local.id).as_str())?;

            let local_info =
                FunctionLocalInfo::new(local_ty, local.ty.clone(), local_alloca, entry_block);
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
        let mut param_types = Vec::new();
        let mut is_variadic = false;

        for ty in extern_.params.iter() {
            match ty {
                mir::Ty::Variadic => {
                    is_variadic = true;
                }
                _ => {
                    let ty = Self::basic_type_to_llvm_basic_metadata_type(self.ctx(), ty);
                    param_types.push(ty);
                }
            }
        }

        let fn_ty = Self::type_to_fn_type(self.ctx(), &extern_.return_ty, param_types, is_variadic);
        let _ = self.module.add_function(&extern_.name, fn_ty, None);
        Ok(())
    }

    fn wrap_main(&self) -> Result<()> {
        let actual_main = self.module.get_function("__entry").unwrap();

        let fn_ty = self.ctx().i32_type().fn_type(&[], false);
        let fn_val = self.module.add_function("main", fn_ty, None);
        let entry_block = self.ctx().append_basic_block(fn_val, "entry");

        self.builder.position_at_end(entry_block);
        self.builder.build_call(actual_main, &[], "_")?;
        self.builder
            .build_return(Some(&self.ctx().i32_type().const_zero()))?;

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

impl<'ctx> Compiler for Llvm<'ctx> {
    fn compile(mut self, module: mir::Module, ctx: FrontendCtx, debug: bool) -> Result<()> {
        for extern_ in module.externs {
            self.compile_extern(extern_)?;
        }

        for function in &module.functions {
            self.precompute_function(function);
        }

        for function in module.functions {
            self.compile_function(function, &ctx)?;
        }

        self.wrap_main()?;

        if debug {
            self.module.print_to_stderr();
        }

        self.module.verify().unwrap();

        self.output();

        Ok(())
    }
}
pub fn compile(module: mir::Module, ctx: FrontendCtx, debug: bool) -> Result<()> {
    let context = Context::create();

    let llvm = Llvm::new(&context);
    llvm.compile(module, ctx, debug)?;

    Ok(())
}
