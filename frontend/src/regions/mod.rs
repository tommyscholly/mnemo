use crate::mir;

pub fn validate_regions(module: &mir::Module) -> Result<(), String> {
    let mut errors = Vec::new();
    for func in &module.functions {
        if let Err(func_errors) = RegionChecker::check(func) {
            errors.extend(func_errors);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        let error_str = errors
            .iter()
            .map(|e| format!("{:?}", e))
            .collect::<Vec<_>>()
            .join("\n");
        Err(error_str)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RegionError {
    pub kind: RegionErrorKind,
    pub function_name: String,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum RegionErrorKind {
    UseAfterRegionEnd {
        local: mir::LocalId,
        region: mir::RegionId,
    },
    RegionEscape {
        local: mir::LocalId,
        local_region: mir::RegionId,
        escapes_to: mir::RegionId,
    },
    LifetimeMismatch {
        source_local: mir::LocalId,
        source_region: mir::RegionId,
        dest_local: mir::LocalId,
        dest_region: mir::RegionId,
    },
}

struct RegionChecker<'a> {
    func: &'a mir::Function,
    region_live: Vec<bool>,
    errors: Vec<RegionError>,
}

impl<'a> RegionChecker<'a> {
    fn check(func: &'a mir::Function) -> Result<(), Vec<RegionError>> {
        let mut checker = Self {
            func,
            region_live: vec![false; Self::max_region_id(func) + 1],
            errors: Vec::new(),
        };
        checker.check_function();
        if checker.errors.is_empty() {
            Ok(())
        } else {
            Err(checker.errors)
        }
    }

    fn max_region_id(func: &mir::Function) -> usize {
        let mut max = 0;
        for block in &func.blocks {
            for stmt in &block.stmts {
                match stmt {
                    mir::Statement::RegionStart(rid) | mir::Statement::RegionEnd(rid) => {
                        max = max.max(*rid);
                    }
                    _ => {}
                }
            }
        }
        max.max(func.region_params.iter().map(|r| r.id).max().unwrap_or(0))
    }

    fn check_function(&mut self) {
        for block in &self.func.blocks {
            self.check_block(block);
        }
    }

    fn check_block(&mut self, block: &mir::BasicBlock) {
        for stmt in &block.stmts {
            self.check_statement(stmt);
        }
        self.check_terminator(&block.terminator);
    }

    fn check_statement(&mut self, stmt: &mir::Statement) {
        match stmt {
            mir::Statement::Assign(local_id, rvalue) => {
                self.check_rvalue(rvalue);
                self.check_local_region(*local_id);
            }
            mir::Statement::Store(place, rvalue) => {
                self.check_place(place);
                self.check_rvalue(rvalue);
            }
            mir::Statement::Phi(dest_local, sources) => {
                self.check_phi(*dest_local, sources);
            }
            mir::Statement::RegionStart(rid) => {
                self.region_live[*rid] = true;
            }
            mir::Statement::RegionEnd(rid) => {
                self.region_live[*rid] = false;
            }
            mir::Statement::Call {
                function_name: _,
                args,
                destination,
            } => {
                for arg in args {
                    self.check_rvalue(arg);
                }
                if let Some(local_id) = destination {
                    self.check_local_region(*local_id);
                }
            }
        }
    }

    fn check_terminator(&mut self, terminator: &mir::Terminator) {
        match terminator {
            mir::Terminator::Return(Some(local_id)) => {
                self.check_local_region(*local_id);
                let local = self.func.locals.iter().find(|l| l.id == *local_id).unwrap();
                if let mir::Ty::Ptr(_, region_id) = &local.ty {
                    if *region_id != mir::STATIC_REGION && !self.is_region_param(*region_id) {
                        self.errors.push(RegionError {
                            kind: RegionErrorKind::RegionEscape {
                                local: *local_id,
                                local_region: *region_id,
                                escapes_to: mir::STATIC_REGION,
                            },
                            function_name: self.func.name.clone(),
                        });
                    }
                }
            }
            mir::Terminator::Br(_) | mir::Terminator::BrIf(_, _, _) => {}
            mir::Terminator::BrTable(_, _) => {}
            _ => {}
        }
    }

    fn check_rvalue(&mut self, rvalue: &mir::RValue) {
        match rvalue {
            mir::RValue::Use(op) => self.check_operand(op),
            mir::RValue::BinOp(_, lhs, rhs) => {
                self.check_operand(lhs);
                self.check_operand(rhs);
            }
            mir::RValue::Alloc {
                kind: _, operands, ..
            } => {
                for op in operands {
                    self.check_operand(op);
                }
            }
        }
    }

    fn check_operand(&mut self, operand: &mir::Operand) {
        match operand {
            mir::Operand::Constant(_) => {}
            mir::Operand::Copy(place) => self.check_place(place),
        }
    }

    fn check_place(&mut self, place: &mir::Place) {
        self.check_local_region(place.local);
    }

    fn check_local_region(&mut self, local_id: mir::LocalId) {
        let local = match self.func.locals.iter().find(|l| l.id == local_id) {
            Some(l) => l,
            None => return,
        };
        if let mir::Ty::Ptr(_, region_id) = &local.ty {
            if !self.region_live[*region_id] && *region_id != mir::STATIC_REGION {
                self.errors.push(RegionError {
                    kind: RegionErrorKind::UseAfterRegionEnd {
                        local: local_id,
                        region: *region_id,
                    },
                    function_name: self.func.name.clone(),
                });
            }
        }
    }

    fn check_phi(&mut self, dest_local: mir::LocalId, sources: &[mir::LocalId]) {
        let dest_local_info = match self.func.locals.iter().find(|l| l.id == dest_local) {
            Some(l) => l,
            None => return,
        };
        let dest_region = match &dest_local_info.ty {
            mir::Ty::Ptr(_, rid) => *rid,
            _ => return,
        };

        for &source_local in sources {
            let source_local_info = match self.func.locals.iter().find(|l| l.id == source_local) {
                Some(l) => l,
                None => continue,
            };
            let source_region = match &source_local_info.ty {
                mir::Ty::Ptr(_, rid) => *rid,
                _ => continue,
            };

            if !self.outlives(source_region, dest_region) {
                self.errors.push(RegionError {
                    kind: RegionErrorKind::LifetimeMismatch {
                        source_local,
                        source_region,
                        dest_local,
                        dest_region,
                    },
                    function_name: self.func.name.clone(),
                });
            }
        }
    }

    fn is_region_param(&self, region_id: mir::RegionId) -> bool {
        self.func.region_params.iter().any(|r| r.id == region_id)
    }

    fn outlives(&self, a: mir::RegionId, b: mir::RegionId) -> bool {
        if a == b {
            return true;
        }
        if a == mir::STATIC_REGION {
            return true;
        }
        self.func
            .region_outlives
            .iter()
            .any(|(outer, inner)| *outer == a && (*inner == b || self.outlives(*inner, b)))
    }
}
