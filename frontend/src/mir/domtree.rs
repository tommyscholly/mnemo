use std::collections::HashMap;

pub trait Searchable<V: Clone> {
    type NodeId: Clone + Eq + std::hash::Hash;

    /// Required methods:
    /// Depth-first search traversal of the graph starting from the given node, applying the visit
    /// function to each node
    fn dfs<F>(&self, start: Self::NodeId, visit: &mut F)
    where
        F: FnMut(Self::NodeId);

    /// Returns the nodes in the graph, order is not guaranteed
    fn nodes(&self) -> Vec<Self::NodeId>;

    /// Derived method:
    /// Collects the nodes in post order starting from the given node
    fn post_order_collect(&self, start: Self::NodeId) -> Vec<Self::NodeId> {
        let mut visited = Vec::new();
        self.dfs(start, &mut |id| {
            visited.push(id.clone());
        });
        visited
    }
}

pub trait DomTreeImpl<V>: Searchable<V>
where
    V: Clone + Eq + std::hash::Hash,
{
    /// Required methods:
    ///
    /// Returns the dominator of a node in the dominator tree
    fn dom(&self, start: Self::NodeId) -> Option<Self::NodeId>;
    /// Sets the dominator of a node in the dominator tree
    fn set_dom(&mut self, node: Self::NodeId, dom: Option<Self::NodeId>);
    /// Clears the dominator tree, which is a precondition for computing the dominator tree
    fn reset_dom(&mut self);
    /// Returns the predecessors of a node in the dominator tree
    fn preds(&self, node: Self::NodeId) -> Vec<Self::NodeId>;

    /// Derived methods:
    /// Helper method to compute the dominance intersection of two nodes
    fn intersect(
        &self,
        order_map: &HashMap<Self::NodeId, usize>,
        mut node1: Self::NodeId,
        mut node2: Self::NodeId,
    ) -> Self::NodeId {
        while node1 != node2 {
            if order_map[&node1] > order_map[&node2] {
                node1 = self
                    .dom(node1.clone())
                    .expect("Every node must eventually have a dominator");
            } else {
                node2 = self
                    .dom(node2.clone())
                    .expect("Every node must eventually have a dominator");
            }
        }
        node1
    }

    /// Computes the dominator tree starting from the given node
    /// https://en.wikipedia.org/wiki/Dominator_(graph_theory)#Algorithms
    fn compute_dom_tree(&mut self, start: Self::NodeId) {
        self.reset_dom();
        self.set_dom(start.clone(), Some(start.clone()));

        let mut order: Vec<Self::NodeId> = self.post_order_collect(start);
        order.reverse();

        let order_map: HashMap<Self::NodeId, usize> = order
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        let mut changed = true;
        while changed {
            changed = false;

            // for each n in N - {n0}:
            for node in order.iter().skip(1) {
                let preds = &self.preds(node.clone());

                let mut new_dominator_opt = None;
                for pred in preds {
                    if self.dom(pred.clone()).is_some() {
                        new_dominator_opt = Some(pred.clone());
                        break;
                    }
                }
                let mut new_dominator = match new_dominator_opt {
                    Some(n) => n,
                    None => continue, // if no predecessor has a computed dominator, skip
                };

                for pred in preds {
                    if self.dom(pred.clone()).is_some() {
                        new_dominator = self.intersect(&order_map, new_dominator, pred.clone());
                    }
                }

                if self.dom(node.clone()) != Some(new_dominator.clone()) {
                    self.set_dom(node.clone(), Some(new_dominator));
                    changed = true;
                }
            }
        }
    }

    /// Returns the dominator tree as a HashMap
    fn dom_tree(&self) -> HashMap<Self::NodeId, Self::NodeId> {
        let mut dom_tree = HashMap::new();
        for node in self.nodes() {
            if let Some(dom) = self.dom(node.clone()) {
                dom_tree.insert(node.clone(), dom);
            }
        }
        dom_tree
    }
}
