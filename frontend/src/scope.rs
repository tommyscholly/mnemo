pub type ScopeId = usize;
pub const ROOT_SCOPE: ScopeId = 0;

#[derive(Debug, Clone)]
pub struct Scope {
    pub parent: Option<ScopeId>,
}

impl Scope {
    pub fn new(parent: Option<ScopeId>) -> Self {
        Self { parent }
    }
}

#[derive(Debug)]
pub struct ScopeTree {
    scopes: Vec<Scope>,
    current: ScopeId,
}

impl ScopeTree {
    pub fn new() -> Self {
        let root_scope = Scope::new(None);
        Self {
            scopes: vec![root_scope],
            current: ROOT_SCOPE,
        }
    }

    pub fn enter(&mut self) -> ScopeId {
        let id = self.scopes.len();
        let scope = Scope::new(Some(self.current));
        self.scopes.push(scope);
        self.current = id;
        id
    }

    pub fn exit(&mut self) {
        if let Some(parent) = self.scopes[self.current].parent {
            self.current = parent;
        }
    }

    pub fn parent_of(&self, id: ScopeId) -> Option<ScopeId> {
        self.scopes.get(id).and_then(|s| s.parent)
    }

    pub fn outlives(&self, outer: ScopeId, inner: ScopeId) -> bool {
        if outer == inner {
            return true;
        }
        let mut current = inner;
        while let Some(parent) = self.parent_of(current) {
            if parent == outer {
                return true;
            }
            current = parent;
        }
        false
    }
}

impl Default for ScopeTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlives_same_scope() {
        let tree = ScopeTree::new();
        assert!(tree.outlives(ROOT_SCOPE, ROOT_SCOPE));
    }

    #[test]
    fn test_root_outlives_child() {
        let mut tree = ScopeTree::new();
        let child = tree.enter();
        assert!(tree.outlives(ROOT_SCOPE, child));
    }

    #[test]
    fn test_child_does_not_outlive_root() {
        let mut tree = ScopeTree::new();
        let child = tree.enter();
        assert!(!tree.outlives(child, ROOT_SCOPE));
    }

    #[test]
    fn test_transitive_outlives() {
        let mut tree = ScopeTree::new();
        let a = tree.enter();
        let b = tree.enter();
        let c = tree.enter();

        assert!(tree.outlives(ROOT_SCOPE, c));
        assert!(tree.outlives(a, c));
        assert!(tree.outlives(b, c));
        assert!(!tree.outlives(c, ROOT_SCOPE));
        assert!(!tree.outlives(c, a));
    }

    #[test]
    fn test_sibling_scopes() {
        let mut tree = ScopeTree::new();
        let a = tree.enter();
        tree.exit();
        let b = tree.enter();

        assert!(!tree.outlives(a, b));
        assert!(!tree.outlives(b, a));
        assert!(tree.outlives(ROOT_SCOPE, a));
        assert!(tree.outlives(ROOT_SCOPE, b));
    }

    #[test]
    fn test_deep_nesting() {
        let mut tree = ScopeTree::new();
        let ids: Vec<ScopeId> = (0..10).map(|_| tree.enter()).collect();

        for i in 0..10 {
            for j in i..10 {
                assert!(
                    tree.outlives(ids[i], ids[j]),
                    "scope {} should outlive {}",
                    i,
                    j
                );
            }
            for j in 0..i {
                assert!(
                    !tree.outlives(ids[i], ids[j]),
                    "scope {} should not outlive {}",
                    i,
                    j
                );
            }
        }
    }
}
