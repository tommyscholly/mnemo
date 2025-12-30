use frontend::{Ctx as FrontendCtx, mir};

use anyhow::Result;
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
use inkwell::values::{IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

use crate::Compiler;

pub struct LLVM<'ctx> {
    module: Module<'ctx>,
    builder: Builder<'ctx>,
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
        }
    }

    fn ctx(&self) -> ContextRef<'ctx> {
        self.module.get_context()
    }

    fn basic_type_to_llvm_basic_type(
        ctx: ContextRef<'ctx>,
        ty: &mir::Ty,
    ) -> BasicMetadataTypeEnum<'ctx> {
        match ty {
            mir::Ty::Int => ctx.i32_type().as_basic_type_enum().into(),
            mir::Ty::Bool => ctx.bool_type().as_basic_type_enum().into(),
            mir::Ty::Char => ctx.i8_type().as_basic_type_enum().into(),
            mir::Ty::Unit => panic!("Unit type not supported as a basic type"),
            _ => todo!(),
        }
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

    fn compile_function(&self, function: mir::Function, ctx: &FrontendCtx) -> Result<()> {
        let mut arg_types = Vec::new();
        let llvm_ctx = self.ctx();

        for (_i, local) in function.locals.iter().enumerate().take(function.parameters) {
            arg_types.push(Self::basic_type_to_llvm_basic_type(llvm_ctx, &local.ty));
        }

        let fn_type = Self::type_to_fn_type(llvm_ctx, &function.return_ty, arg_types);
        self.module.add_function(
            ctx.resolve(Symbol(function.function_id)),
            fn_type,
            Some(Linkage::External),
        );

        Ok(())
    }
}

impl<'ctx> Compiler for LLVM<'ctx> {
    fn compile(self, module: mir::Module, ctx: FrontendCtx) -> Result<()> {
        for function in module.functions {
            self.compile_function(function, &ctx)?;
        }
        Ok(())
    }
}
pub fn compile(module: mir::Module, ctx: FrontendCtx) -> Result<()> {
    let context = Context::create();

    let llvm = LLVM::new(&context);
    llvm.compile(module, ctx)?;

    Ok(())
}
