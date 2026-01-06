#![allow(unused)]

use std::collections::{HashMap, HashSet};

use crate::{
    Ctx,
    ast::{
        self, AllocKind, Block, BlockInner, Call, Decl, DeclKind, Expr, ExprKind, IfElse, Match,
        Module, Params, Pat, PatKind, Region, Signature, SignatureInner, Stmt, StmtKind, Type,
        TypeKind, ValueKind, VariantField,
    },
    ctx::Symbol,
    span::{DUMMY_SPAN, Diagnostic, Span, Spanned},
};

#[derive(Debug)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
}

impl TypeError {
    fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl Diagnostic for TypeError {
    fn span(&self) -> &Span {
        &self.span
    }

    fn message(&self) -> String {
        format!("type error {:?}", self.kind)
    }

    fn label(&self) -> Option<String> {
        Some(format!("{:?}", self.kind))
    }
}

#[derive(Debug)]
pub enum TypeErrorKind {
    MissingMainFunction,
    ExpectedArrayIndex,
    ExpectedVariant,
    ExpectedType {
        expected: TypeKind,
        found: TypeKind,
    },
    VariantMismatch {
        expected: Vec<VariantField>,
        found: VariantField,
    },
    FnTypeExpected,
    SignatureMismatch {
        expected: Box<SignatureInner>,
        found: Box<SignatureInner>,
    },
    RecordFieldMissing {
        field_name: Symbol,
        declared_fields: Vec<Symbol>,
    },
    RecordFieldMismatch {
        field_name: Symbol,
        expected: TypeKind,
        found: TypeKind,
    },
    ReturnTypeMismatch {
        expected: TypeKind,
        found: TypeKind,
    },
    UnknownSymbol(Symbol),
    UnknownField(Symbol),
    ArgCountMismatch {
        expected: usize,
        found: usize,
    },
    UnknownMethod(Symbol),
    NotCallable,
    ExpectedRecord,
}

pub struct TypecheckCtx<'a> {
    front_ctx: &'a mut Ctx,
    // TOOD: track scope of symbols
    type_map: HashMap<Symbol, Type>,
    function_sigs: HashMap<Symbol, Signature>,
}

impl<'a> TypecheckCtx<'a> {
    fn new(ctx: &'a mut Ctx) -> Self {
        let mut function_sigs = HashMap::new();

        Self {
            front_ctx: ctx,
            type_map: HashMap::new(),
            function_sigs,
        }
    }
}

pub type TypecheckResult<T> = Result<T, TypeError>;

trait ResolveType {
    fn resolve_type(&self, ctx: &TypecheckCtx) -> Type;
}

impl ResolveType for Expr {
    fn resolve_type(&self, ctx: &TypecheckCtx) -> Type {
        match &self.node {
            ExprKind::Value(v) => match v {
                ValueKind::Int(_) => Type::synthetic(TypeKind::Int),
                ValueKind::Ident(i) => ctx.type_map.get(i).unwrap().clone(),
                ValueKind::Bool(_) => Type::synthetic(TypeKind::Bool),
            },
            ExprKind::BinOp { lhs, rhs, .. } => {
                let lhs_ty = lhs.resolve_type(ctx);
                let rhs_ty = rhs.resolve_type(ctx);
                if lhs_ty.node != rhs_ty.node {
                    panic!("expected types to be equal");
                }
                lhs_ty
            }
            ExprKind::Call(Call {
                callee,
                args: _,
                returned_ty,
            }) => {
                let (callee_sig, _) = resolve_callee_signature(callee, ctx).unwrap();
                returned_ty.clone().unwrap()
            }
            ExprKind::Allocation {
                kind,
                region,
                elements,
            } => {
                // TODO: implement region handling
                // regions should be resolved by the time we get here
                let region_handle = region.unwrap_or(Region::Stack);
                let kind = match kind {
                    AllocKind::Tuple(tys) => {
                        let mut types = Vec::new();
                        for elem in elements {
                            types.push(elem.resolve_type(ctx).node);
                        }

                        if !tys.is_empty() && tys.len() != types.len() && *tys != types {
                            panic!("expected tuple types to be equal");
                        }

                        // here we resolve the tuple types if they were not provided by a type hint
                        AllocKind::Tuple(types)
                    }
                    AllocKind::Variant(variant_name) => {
                        let adts = elements.iter().map(|e| e.resolve_type(ctx)).collect();

                        let ty = TypeKind::Variant(vec![VariantField {
                            name: Spanned::default(*variant_name),
                            adts,
                        }]);

                        return Type::synthetic(ty);
                    }
                    AllocKind::Record(fields) => {
                        return Type::synthetic(TypeKind::Record(fields.clone()));
                    }
                    _ => kind.clone(),
                };

                Type::synthetic(TypeKind::Alloc(kind, region_handle))
            }
            ExprKind::FieldAccess(expr, field) => {
                let expr_ty = expr.resolve_type(ctx);
                let field_ty = get_field_type(&expr_ty, *field).unwrap();

                Type::synthetic(field_ty)
            }
            ExprKind::TupleAccess(expr, index) => {
                let expr_ty = expr.resolve_type(ctx);

                let TypeKind::Alloc(AllocKind::Tuple(tys), _) = expr_ty.node else {
                    panic!("expected tuple type, got {:?}", expr_ty.node);
                };

                if *index >= tys.len() {
                    panic!("tuple index out of bounds");
                }

                let ty = tys[*index].clone();

                Type::synthetic(ty)
            }
            ExprKind::Index(expr, index) => {
                let expr_ty = expr.resolve_type(ctx);

                let TypeKind::Alloc(AllocKind::Array(ty, _), _) = expr_ty.node else {
                    panic!("expected array type, got {:?}", expr_ty.node);
                };

                Type::with_span(*ty, expr.span.clone())
            }
        }
    }
}

trait Typecheck {
    // INVARIANT: After typechecking, all types are either fully resolved, or an error is returned.
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()>;
}

fn type_check_call(call: &mut Call, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
    for arg in call.args.iter_mut() {
        arg.typecheck(ctx)?;
    }

    let (callee_signature, span) = resolve_callee_signature(&call.callee, ctx)?;

    let expected_count = callee_signature.node.params.types.len();
    let found_count = call.args.len();
    if expected_count != found_count {
        return Err(TypeError::new(
            TypeErrorKind::ArgCountMismatch {
                expected: expected_count,
                found: found_count,
            },
            call.callee.span.clone(),
        ));
    }

    for (arg, ty) in call
        .args
        .iter()
        .zip(callee_signature.node.params.types.iter())
    {
        let arg_ty = arg.resolve_type(ctx);
        structural_typecheck(&arg_ty.node, &ty.node, arg_ty.span)?;
    }

    if let Some(returned_ty) = &call.returned_ty {
        if callee_signature.node.return_ty.is_none() {
            return Err(TypeError::new(
                TypeErrorKind::ReturnTypeMismatch {
                    expected: TypeKind::Unit,
                    found: returned_ty.node.clone(),
                },
                call.callee.span.clone(),
            ));
        }

        if callee_signature.node.return_ty != Some(returned_ty.clone()) {
            return Err(TypeError::new(
                TypeErrorKind::ReturnTypeMismatch {
                    expected: callee_signature.node.return_ty.clone().unwrap().node,
                    found: returned_ty.node.clone(),
                },
                call.callee.span.clone(),
            ));
        }
    } else {
        match callee_signature.node.return_ty {
            Some(ty) => {
                call.returned_ty = Some(ty);
            }
            None => {
                call.returned_ty = Some(Type::synthetic(TypeKind::Unit));
            }
        }
    }

    Ok(())
}

pub fn resolve_callee_signature(
    callee: &Expr,
    ctx: &TypecheckCtx,
) -> TypecheckResult<(Signature, Span)> {
    match &callee.node {
        ExprKind::Value(ValueKind::Ident(name)) => {
            let Some(sig) = ctx.function_sigs.get(name) else {
                return Err(TypeError::new(
                    TypeErrorKind::UnknownSymbol(*name),
                    callee.span.clone(),
                ));
            };
            Ok((sig.clone(), callee.span.clone()))
        }
        // for now, we are doing UFCS: x.method() = method(x)
        // eventually we may have structural typing impls
        ExprKind::FieldAccess(receiver, method_name) => {
            let Some(sig) = ctx.function_sigs.get(method_name) else {
                return Err(TypeError::new(
                    TypeErrorKind::UnknownMethod(*method_name),
                    callee.span.clone(),
                ));
            };
            Ok((sig.clone(), callee.span.clone()))
        }
        _ => {
            // For first-class functions, you'd check that callee's type is TypeKind::Fn
            // and extract the signature from there
            Err(TypeError::new(
                TypeErrorKind::NotCallable,
                callee.span.clone(),
            ))
        }
    }
}

fn get_field_type(expr_ty: &Type, field: Symbol) -> TypecheckResult<TypeKind> {
    let type_kind = &expr_ty.node;
    let field_ty = if let TypeKind::Record(fields) = &expr_ty.node {
        fields
            .iter()
            .find(|f| f.name == field)
            // SAFETY: field types should be resolved by now
            .map(|f| f.ty.as_ref().unwrap())
    } else {
        return Err(TypeError::new(
            TypeErrorKind::ExpectedRecord,
            expr_ty.span.clone(),
        ));
    };

    match field_ty {
        Some(ty) => Ok(ty.node.clone()),
        None => Err(TypeError::new(
            TypeErrorKind::UnknownField(field),
            expr_ty.span.clone(),
        )),
    }
}

impl Typecheck for Expr {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match &mut self.node {
            ExprKind::Value(v) => match v {
                ValueKind::Int(_) => Ok(()),
                ValueKind::Bool(_) => Ok(()),
                ValueKind::Ident(i) => {
                    if !ctx.type_map.contains_key(i) {
                        return Err(TypeError::new(
                            TypeErrorKind::UnknownSymbol(*i),
                            self.span.clone(),
                        ));
                    }
                    Ok(())
                }
            },
            ExprKind::BinOp { lhs, rhs, .. } => {
                lhs.typecheck(ctx)?;
                rhs.typecheck(ctx)?;
                Ok(())
            }
            ExprKind::Call(c) => type_check_call(c, ctx),
            ExprKind::Allocation { elements, kind, .. } => {
                for e in &mut *elements {
                    e.typecheck(ctx)?;
                }

                match kind {
                    AllocKind::Tuple(tys) => {
                        let mut types = Vec::new();
                        for elem in elements {
                            types.push(elem.resolve_type(ctx).node);
                        }

                        if !tys.is_empty() && tys.len() != types.len() && *tys != types {
                            panic!("expected tuple types to be equal");
                        }

                        *tys = types;
                    }
                    AllocKind::Array(ty_kind, size) => {
                        assert_eq!(*size, elements.len());

                        assert!(
                            elements
                                .iter()
                                .all(|e| e.resolve_type(ctx).node == **ty_kind)
                        );
                    }
                    AllocKind::DynArray(ty_kind) => {
                        assert!(
                            elements
                                .iter()
                                .all(|e| e.resolve_type(ctx).node == **ty_kind)
                        );
                    }
                    AllocKind::Record(fields) => {
                        assert_eq!(elements.len(), fields.len());
                        for (field, elem) in fields.iter_mut().zip(elements.iter()) {
                            let elem_ty = elem.resolve_type(ctx);

                            match &field.ty {
                                Some(ty) => {
                                    assert_eq!(elem_ty.node, ty.node);
                                }
                                None => {
                                    field.ty = Some(elem_ty);
                                }
                            }
                        }
                    }
                    AllocKind::Variant(variant_name) => {
                        // TOOD: is there any variant to check here?
                    }
                    AllocKind::Str(_) => {}
                    _ => todo!(),
                }
                Ok(())
            }
            ExprKind::FieldAccess(expr, field) => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);
                let _field_ty = get_field_type(&expr_ty, *field)?;

                Ok(())
            }
            ExprKind::Index(expr, index) => {
                expr.typecheck(ctx)?;
                index.typecheck(ctx)?;

                let expr_ty = expr.resolve_type(ctx).node;
                let index_ty = index.resolve_type(ctx).node;

                match (expr_ty, index_ty) {
                    (TypeKind::Alloc(AllocKind::Array(_, _), _), TypeKind::Int) => {}
                    _ => {
                        return Err(TypeError::new(
                            TypeErrorKind::ExpectedArrayIndex,
                            index.span.clone(),
                        ));
                    }
                }

                Ok(())
            }
            ExprKind::TupleAccess(expr, index) => {
                expr.typecheck(ctx)?;

                let expr_ty = expr.resolve_type(ctx).node;

                match (expr_ty) {
                    TypeKind::Alloc(AllocKind::Tuple(_), _) => {}
                    _ => {
                        return Err(TypeError::new(
                            TypeErrorKind::ExpectedArrayIndex,
                            expr.span.clone(),
                        ));
                    }
                }

                Ok(())
            }
        }
    }
}

fn bind_pattern(pat: &Pat, ty: &Type, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
    match &pat.node {
        PatKind::Symbol(name) => {
            ctx.type_map.insert(*name, ty.clone());
        }
        _ => todo!(),
    }

    Ok(())
}

fn structural_typecheck(
    expr_ty: &TypeKind,
    declared_ty: &TypeKind,
    declared_ty_span: Span,
) -> TypecheckResult<()> {
    match (expr_ty, declared_ty) {
        (TypeKind::Variant(expr_variant), TypeKind::Variant(declared_variants)) => {
            // SAFETY: expr variants will only have one element
            let expr_variant = expr_variant.first().unwrap();
            if !declared_variants.contains(expr_variant) {
                return Err(TypeError::new(
                    TypeErrorKind::VariantMismatch {
                        expected: declared_variants.clone(),
                        found: expr_variant.clone(),
                    },
                    declared_ty_span,
                ));
            }
        }
        // we ensure that the declared fields, which is the type annotation, is a subset of the
        // actual expression fields
        // for instance, if we have a function that takes in a record { a: int, b: int }, the expr
        // fields may be { a: int, b: int, c: int } but could not be { a: int }
        // expr <: declared_fields, or in this example, { a: int, b: int, c: int } <: { a: int, b: int }
        (TypeKind::Record(expr_fields), TypeKind::Record(declared_fields)) => {
            let expr_field_set: HashMap<u32, TypeKind> = expr_fields
                .iter()
                .map(|f| (f.name.0, f.ty.as_ref().unwrap().node.clone()))
                .collect();

            let declared_field_set: HashMap<u32, TypeKind> = declared_fields
                .iter()
                .map(|f| (f.name.0, f.ty.as_ref().unwrap().node.clone()))
                .collect();

            // ensure that all the declared fields are present in the expression
            for (field_name, field_ty) in &declared_field_set {
                if !expr_field_set.contains_key(field_name) {
                    return Err(TypeError::new(
                        TypeErrorKind::RecordFieldMissing {
                            field_name: Symbol(*field_name),
                            declared_fields: declared_field_set
                                .keys()
                                .cloned()
                                .map(Into::into)
                                .collect(),
                        },
                        declared_ty_span,
                    ));
                }

                if expr_field_set[field_name] != declared_field_set[field_name] {
                    return Err(TypeError::new(
                        TypeErrorKind::RecordFieldMismatch {
                            field_name: Symbol(*field_name),
                            expected: declared_field_set[field_name].clone(),
                            found: field_ty.clone(),
                        },
                        declared_ty_span,
                    ));
                }
            }
        }
        (TypeKind::Alloc(AllocKind::Str(_), _), TypeKind::Ptr(c)) if (**c) == TypeKind::Char => {}
        _ => {
            if declared_ty != expr_ty {
                return Err(TypeError::new(
                    TypeErrorKind::ExpectedType {
                        expected: declared_ty.clone(),
                        found: expr_ty.clone(),
                    },
                    declared_ty_span,
                ));
            }
        }
    }

    Ok(())
}

impl Typecheck for Stmt {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match &mut self.node {
            StmtKind::ValDec { name, ty, expr } => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);

                if let Some(declared_ty) = ty {
                    structural_typecheck(
                        &expr_ty.node,
                        &declared_ty.node,
                        declared_ty.span.clone(),
                    )?;

                    ctx.type_map.insert(name.node, declared_ty.clone());
                } else {
                    *ty = Some(expr_ty.clone());
                    ctx.type_map.insert(name.node, expr_ty);
                }
            }
            StmtKind::Assign { name, expr } => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);

                let Some(expected_type) = ctx.type_map.get(&name.node) else {
                    return Err(TypeError::new(
                        TypeErrorKind::UnknownSymbol(name.node),
                        name.span.clone(),
                    ));
                };

                if expected_type.node != expr_ty.node {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: expected_type.node.clone(),
                            found: expr_ty.node,
                        },
                        expr.span.clone(),
                    ));
                }
            }
            StmtKind::IfElse(IfElse { cond, then, else_ }) => {
                // TODO: check that cond is a bool, right now we only have ints
                cond.typecheck(ctx)?;
                then.typecheck(ctx)?;
                if let Some(else_) = else_ {
                    else_.typecheck(ctx)?;
                }
            }
            StmtKind::Call(c) => type_check_call(c, ctx)?,
            StmtKind::Match(Match { scrutinee, arms }) => {
                scrutinee.typecheck(ctx)?;
                let scrutinee_ty = scrutinee.resolve_type(ctx);
                for arm in arms {
                    match &arm.pat.node {
                        PatKind::Symbol(name) => {
                            // symbols shouldnt need type checking, we just have to bind them in the
                            // type map
                            bind_pattern(&arm.pat, &scrutinee_ty, ctx)?;
                        }
                        PatKind::Variant { name, bindings } => {
                            let TypeKind::Variant(vfields) = &scrutinee_ty.node else {
                                return Err(TypeError::new(
                                    TypeErrorKind::ExpectedVariant,
                                    scrutinee_ty.span.clone(),
                                ));
                            };

                            let Some(variant) = vfields.iter().find(|v| v.name.node == *name)
                            else {
                                return Err(TypeError::new(
                                    TypeErrorKind::UnknownSymbol(*name),
                                    arm.pat.span.clone(),
                                ));
                            };

                            if bindings.len() != variant.adts.len() {
                                return Err(TypeError::new(
                                    // TODO: better error message
                                    TypeErrorKind::ExpectedVariant,
                                    arm.pat.span.clone(),
                                ));
                            }

                            for (binding, adt) in bindings.iter().zip(variant.adts.iter()) {
                                bind_pattern(binding, adt, ctx)?;
                            }
                        }
                        PatKind::Record(fields) => {
                            let TypeKind::Record(scru_fields) = &scrutinee_ty.node else {
                                return Err(TypeError::new(
                                    TypeErrorKind::ExpectedRecord,
                                    scrutinee_ty.span.clone(),
                                ));
                            };

                            let pat_ty = TypeKind::Record(fields.clone());
                            structural_typecheck(
                                &scrutinee_ty.node,
                                &pat_ty,
                                scrutinee_ty.span.clone(),
                            )?;

                            for field in fields {
                                ctx.type_map.insert(field.name, field.ty.clone().unwrap());
                            }
                        }
                        PatKind::Literal(lit) => match lit {
                            ValueKind::Int(_) => {
                                if scrutinee_ty.node != TypeKind::Int {
                                    return Err(TypeError::new(
                                        TypeErrorKind::ExpectedType {
                                            expected: TypeKind::Int,
                                            found: scrutinee_ty.node.clone(),
                                        },
                                        scrutinee_ty.span.clone(),
                                    ));
                                }
                            }
                            _ => todo!(),
                        },
                        PatKind::Wildcard => {}
                        _ => todo!(),
                    }

                    arm.body.typecheck(ctx)?;
                }
            }
            StmtKind::Return(expr) => {
                if let Some(expr) = expr {
                    expr.typecheck(ctx)?;
                }
            }
        }

        Ok(())
    }
}

impl Typecheck for Block {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        for stmt in self.node.stmts.iter_mut() {
            stmt.typecheck(ctx)?;
        }

        if let Some(expr) = &mut self.node.expr {
            expr.typecheck(ctx)?;
        }

        Ok(())
    }
}

impl Typecheck for Decl {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match &mut self.node {
            DeclKind::Extern { name, sig } => {
                ctx.function_sigs.insert(name.node, sig.clone());
            }
            DeclKind::Constant { name, ty, expr } => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);

                if let Some(declared_ty) = ty {
                    if declared_ty.node != expr_ty.node {
                        return Err(TypeError::new(
                            TypeErrorKind::ExpectedType {
                                expected: declared_ty.node.clone(),
                                found: expr_ty.node,
                            },
                            expr.span.clone(),
                        ));
                    }
                } else {
                    *ty = Some(expr_ty);
                }
            }
            DeclKind::TypeDef { name: _, def: _ } => {}
            DeclKind::Procedure {
                name,
                fn_ty,
                sig,
                block,
                constraints: _,
            } => {
                if let Some(declared_fn_ty) = fn_ty {
                    let TypeKind::Fn(fn_sig) = &declared_fn_ty.node else {
                        return Err(TypeError::new(
                            TypeErrorKind::FnTypeExpected,
                            declared_fn_ty.span.clone(),
                        ));
                    };

                    if sig.node != **fn_sig {
                        return Err(TypeError::new(
                            TypeErrorKind::SignatureMismatch {
                                expected: fn_sig.clone(),
                                found: Box::new(sig.node.clone()),
                            },
                            sig.span.clone(),
                        ));
                    }
                } else {
                    *fn_ty = Some(Type::synthetic(TypeKind::Fn(Box::new(sig.node.clone()))));
                }

                for (pat, ty) in sig
                    .node
                    .params
                    .patterns
                    .iter()
                    .zip(sig.node.params.types.iter())
                {
                    #[allow(irrefutable_let_patterns)]
                    if let PatKind::Symbol(sym) = pat.node {
                        ctx.type_map.insert(sym, ty.clone());
                    }
                }

                ctx.function_sigs.insert(name.node, sig.clone());
                block.typecheck(ctx)?;

                if let Some(expr) = &block.node.expr
                    && let Some(ret_ty) = &sig.node.return_ty
                {
                    let expr_ty = expr.resolve_type(ctx);

                    if expr_ty.node != ret_ty.node {
                        return Err(TypeError::new(
                            TypeErrorKind::ReturnTypeMismatch {
                                expected: ret_ty.node.clone(),
                                found: expr_ty.node,
                            },
                            expr.span.clone(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

impl Typecheck for Module {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        for decl in self.declarations.iter_mut() {
            decl.typecheck(ctx)?;
        }

        let mut found_main = false;
        for function in ctx.function_sigs.keys() {
            let function_name = ctx.front_ctx.resolve(*function);

            if function_name == "main" {
                ctx.front_ctx.update(*function, "__entry");
                found_main = true;
            }
        }

        #[cfg(not(test))]
        if !found_main {
            return Err(TypeError::new(
                TypeErrorKind::MissingMainFunction,
                DUMMY_SPAN,
            ));
        }

        Ok(())
    }
}

pub fn typecheck(ctx: &mut Ctx, module: &mut Module) -> TypecheckResult<()> {
    let mut ctx = TypecheckCtx::new(ctx);
    module.typecheck(&mut ctx)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::{
        ast::{Pat, PatKind, TypeKind},
        ctx::Ctx,
        lex, parse,
        span::Spanned,
    };

    use super::*;

    fn tokenify(s: &str) -> (Ctx, VecDeque<Spanned<crate::lex::Token>>) {
        let mut ctx = Ctx::default();
        let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
        (ctx, tokens)
    }

    fn parseify(ctx: &mut Ctx, tokens: VecDeque<Spanned<crate::lex::Token>>) -> Module {
        parse::parse(ctx, tokens).unwrap()
    }

    fn typecheck_src(src: &str) -> TypecheckResult<Module> {
        let (mut ctx, tokens) = tokenify(src);
        let mut module = parseify(&mut ctx, tokens);
        typecheck(&mut ctx, &mut module)?;
        Ok(module)
    }

    #[test]
    fn test_typecheck_simple_procedure() {
        let src = "foo :: (x: int): int { y : int = x }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_infers_val_type() {
        let src = "foo :: (x: int): int { y := x }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { ty, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_binop_expression() {
        let src = "foo :: (x: int, y: int): int { z : int = x + y }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_assignment() {
        let src = "foo :: (x: int): int { y : int = x \n y = x + 1 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_if_else() {
        let src = "foo :: (x: int): int { if x { y : int = 1 } else { z : int = 2 } }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_constant() {
        let src = "MY_CONST :: 42";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
            panic!("expected constant");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_function_call() {
        let src = "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { bar(1) }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_function_call_with_return() {
        let src = "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { result : int = bar(1) }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_ident() {
        let src = "foo :: (): int { y : int = unknown }";
        let result = typecheck_src(src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_assignment() {
        let src = "foo :: (): int { y : int = 1 \n y = unknown }";
        let result = typecheck_src(src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_error_unknown_function() {
        let src = "foo :: (): int { unknown_func(1) }";
        let result = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_nested_if() {
        let src = "foo :: (x: int): int { if 1 { if 1 { y : int = 1 } } }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_multiple_params() {
        let src = "foo :: (a: int, b: int, c: int): int { sum : int = 1 + 2 + 3 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_procedure_infers_fn_type() {
        let src = "foo :: (x: int): int { y : int = 1 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { fn_ty, sig, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        assert!(fn_ty.is_some());
        let TypeKind::Fn(fn_sig) = &fn_ty.as_ref().unwrap().node else {
            panic!("expected fn type");
        };
        assert_eq!(**fn_sig, sig.node);
    }

    #[test]
    fn test_typecheck_multiple_declarations() {
        let src = "CONST :: 10 \n bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { bar(1) }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.declarations.len(), 3);
    }

    #[test]
    fn test_typecheck_constant_with_type_annotation() {
        let src = "MY_CONST : int : 42";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
            panic!("expected constant");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_constant_with_binop() {
        let src = "MY_CONST :: 1 + 2 * 3";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
            panic!("expected constant");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_empty_procedure() {
        let src = "foo :: () {}";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_procedure_with_return_type() {
        let src = "foo :: (): int {}";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { sig, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        assert_eq!(
            sig.node.return_ty.as_ref().map(|t| &t.node),
            Some(&TypeKind::Int)
        );
    }

    #[test]
    fn test_typecheck_val_dec_infers_int_from_literal() {
        let src = "foo :: (): int { y := 42 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { ty, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_multiple_val_decs() {
        let src = "foo :: (): int { a : int = 1 \n b : int = 2 \n c : int = 3 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_val_dec_uses_previous_val() {
        let src = "foo :: (): int { a : int = 1 \n b : int = a }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_assignment_to_declared_var() {
        let src = "foo :: (): int { a : int = 1 \n a = 2 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_error_assignment_to_undeclared_var() {
        let src = "foo :: (): int { a = 1 }";
        let result = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_variant_type() {
        let src = "foo :: (): int { x := None }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert!(matches!(
            &expr.node,
            ExprKind::Allocation {
                kind: AllocKind::Variant(variant_name),
                elements,..

            } if *variant_name == Symbol(2) && elements.is_empty()
        ));
    }

    #[test]
    fn test_typecheck_variant_type_with_args() {
        let src = "foo :: (): int { x : Some(int) | None = Some(1) }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert!(matches!(
            &expr.node,
            ExprKind::Allocation {
                kind: AllocKind::Variant(variant_name),
                elements,..

            } if *variant_name == Symbol(2) && elements.len() == 1 && elements[0].node == ExprKind::Value(ValueKind::Int(1))
        ));
    }

    #[test]
    fn test_typecheck_subtyping_record() {
        let src = "foo :: () { x : { a: int } = { a := 1, b := 2 } }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }
}
