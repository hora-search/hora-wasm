import * as hora_wasm from '../pkg/hora_wasm.js';


const demo = () => {
  const dimension = 50;

  var bf_idx = hora_wasm.BruteForceIndexUsize.new(dimension);
  for (var i = 0; i < 1000; i++) {
    var feature = [];
    for (var j = 0; j < dimension; j++) {
      feature.push(Math.random());
    }
    bf_idx.add(feature, i); // add point 
  }
  bf_idx.build("euclidean"); // build index
  var feature = [];
  for (var j = 0; j < dimension; j++) {
    feature.push(Math.random());
  }
  console.log("bf result", bf_idx.search(feature, 10)); //bf result Uint32Array(10) [704, 113, 358, 835, 408, 379, 117, 414, 808, 826]
}

demo();
console.log(hora_wasm)
hora_wasm.greet();