extern crate cfg_if;
extern crate wasm_bindgen;

use cfg_if::cfg_if;
use real_hora::core::ann_index::ANNIndex;

use real_hora::core::metrics;

use wasm_bindgen::prelude::*;
// use rayon::prelude::*;
// pub use wasm_bindgen_rayon::init_thread_pool;


cfg_if! {
    if #[cfg(feature = "wee_alloc")] {
        extern crate wee_alloc;
        #[global_allocator]
        static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
    }
}

fn metrics_transform(s: &str) -> metrics::Metric {
    match s {
        "angular" => metrics::Metric::Angular,
        "manhattan" => metrics::Metric::Manhattan,
        "dot_product" => metrics::Metric::DotProduct,
        "euclidean" => metrics::Metric::Euclidean,
        "cosine_similarity" => metrics::Metric::CosineSimilarity,
        _ => metrics::Metric::Unknown,
    }
}

#[macro_export]
macro_rules! inherit_ann_index_method {
    (  $ann_idx:ident,$type_expr: ty, $idx_type_expr: ty) => {
        #[wasm_bindgen]
        pub struct $ann_idx {
            _idx: Box<$type_expr>,
        }

        #[wasm_bindgen]
        impl $ann_idx {
            pub fn build(&mut self, s: String) -> bool {
                self._idx.build(metrics_transform(&s)).unwrap();
                true
            }
            pub fn add(&mut self, vs: &[f32], idx: $idx_type_expr) -> bool {
                println!("{:?} {:?}", vs, idx);
                self._idx.add(&vs, idx).unwrap();
                true
            }

            pub fn search(&self, vs: &[f32], k: usize) -> Vec<$idx_type_expr> {
                self._idx.search(&vs, k)
            }

            pub fn name(&self) -> String {
                self._idx.name().to_string()
            }
        }
    };
}

inherit_ann_index_method!(BruteForceIndexUsize, real_hora::index::bruteforce_idx::BruteForceIndex<f32,usize>, usize);
#[wasm_bindgen]
impl BruteForceIndexUsize {
    pub fn new(dimension: usize) -> Self {
        BruteForceIndexUsize {
            _idx: Box::new(real_hora::index::bruteforce_idx::BruteForceIndex::<
                f32,
                usize,
            >::new(
                dimension,
                &real_hora::index::bruteforce_params::BruteForceParams::default(),
            )),
        }
    }
}

inherit_ann_index_method!(HNSWIndexUsize, real_hora::index::hnsw_idx::HNSWIndex<f32, usize>,usize);
#[wasm_bindgen]
impl HNSWIndexUsize {

    pub fn new(
        dimension: usize,
        max_item: usize,
        n_neigh: usize,
        n_neigh0: usize,
        ef_build: usize,
        ef_search: usize,
        has_deletion: bool,
    ) -> Self {
        HNSWIndexUsize {
            _idx: Box::new(real_hora::index::hnsw_idx::HNSWIndex::<f32, usize>::new(
                dimension,
                &real_hora::index::hnsw_params::HNSWParams::<f32>::default()
                    .max_item(max_item)
                    .n_neighbor(n_neigh)
                    .n_neighbor0(n_neigh0)
                    .ef_build(ef_build)
                    .ef_search(ef_search)
                    .has_deletion(has_deletion),
            )),
        }
    }
}

inherit_ann_index_method!(PQIndexUsize, real_hora::index::pq_idx::PQIndex<f32, usize>,usize);
#[wasm_bindgen]
impl PQIndexUsize {

    pub fn new(dimension: usize, n_sub: usize, sub_bits: usize, train_epoch: usize) -> Self {
        PQIndexUsize {
            _idx: Box::new(real_hora::index::pq_idx::PQIndex::<f32, usize>::new(
                dimension,
                &real_hora::index::pq_params::PQParams::default()
                    .n_sub(n_sub)
                    .sub_bits(sub_bits)
                    .train_epoch(train_epoch),
            )),
        }
    }
}

inherit_ann_index_method!(IVFPQIndexUsize, real_hora::index::pq_idx::IVFPQIndex<f32, usize>,usize);
#[wasm_bindgen]
impl IVFPQIndexUsize {
    pub fn new(
        dimension: usize,
        n_sub: usize,
        sub_bits: usize,
        n_kmeans_center: usize,
        search_n_center: usize,
        train_epoch: usize,
    ) -> Self {
        IVFPQIndexUsize {
            _idx: Box::new(real_hora::index::pq_idx::IVFPQIndex::<f32, usize>::new(
                dimension,
                &real_hora::index::pq_params::IVFPQParams::default()
                    .n_sub(n_sub)
                    .sub_bits(sub_bits)
                    .n_kmeans_center(n_kmeans_center)
                    .search_n_center(search_n_center)
                    .train_epoch(train_epoch),
            )),
        }
    }
}

inherit_ann_index_method!(SSGIndexUsize, real_hora::index::ssg_idx::SSGIndex<f32, usize>,usize);
#[wasm_bindgen]
impl SSGIndexUsize {
    pub fn new(
        dimension: usize,
        neighbor_neighbor_size: usize,
        init_k: usize,
        index_size: usize,
        angle: f32,
        root_size: usize,
    ) -> Self {
        SSGIndexUsize {
            _idx: Box::new(real_hora::index::ssg_idx::SSGIndex::<f32, usize>::new(
                dimension,
                &real_hora::index::ssg_params::SSGParams::default()
                    .neighbor_neighbor_size(neighbor_neighbor_size)
                    .init_k(init_k)
                    .index_size(index_size)
                    .angle(angle)
                    .root_size(root_size),
            )),
        }
    }
}
