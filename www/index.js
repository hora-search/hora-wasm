import * as hora_wasm from '../pkg/hora_wasm.js';


const demo = () => {
  const dimension = 50;

  // var hnsw_idx = hora_wasm.HNSWIndexUsize.new(dimension, 10000000, 16, 32, 500, 16, false);
  var hnsw_idx = hora_wasm.BruteForceIndexUsize.new(dimension);
  for (var i = 0; i < 1000; i++) {
    var feature = [];
    for (var j = 0; j < dimension; j++) {
      feature.push(Math.random());
    }
    hnsw_idx.add(feature, i);
  }
  hnsw_idx.build("euclidean");
  var feature = [];
  for (var j = 0; j < dimension; j++) {
    feature.push(Math.random());
  }
  console.log("hnsw result", hnsw_idx.search(feature, 10));
}

demo();
console.log(hora_wasm)
hora_wasm.greet();