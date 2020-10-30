alert("This is an untrained neural network. To train it, draw a couple of smiley faces and press the :) button, then draw a couple of frowney faces and press the :( button. After you've done that, press the train button to train it. Once trained, you can have the neural network guess the output of your drawing by drawing either a smiley face or frowney face, then pressing guess!");

class Neural_Network {
  constructor(structure=[]) {
    this.structure = structure;
    this.weight_matrices = [];
    this.layers = [];
    this.gradient_data = [];
    this.gradient_sums = [];
    this.loss = 0;
  }

  setup(optimizer) {
    if (optimizer === "rprop") {
      this.optimizer = "rprop";
      this.prev_weight_matrices = [];
      this.prev_gradient_signs = [];
      this.curr_gradient = [];
      this.alpha = [];
    }
    this.gradient_sums = this.zeroV(this.structure.length);
    for (let i = 0; i < this.structure.length; i++) {
      this.gradient_data[i] = this.zeroV(this.structure[this.structure.length-1-i].nodes);
      if (i) {
        this.weight_matrices[i] = this.randM(this.structure[i].nodes, this.structure[i-1].nodes+1);
        if (optimizer === "rprop") {
          this.prev_weight_matrices[i] = this.randM(this.structure[i].nodes, this.structure[i-1].nodes+1);
          this.prev_gradient_signs[i] = this.zeroM(this.structure[i].nodes, this.structure[i-1].nodes+1);
          this.curr_gradient[i] = this.zeroM(this.structure[i].nodes, this.structure[i-1].nodes+1);
          this.alpha[i] = this.zeroM(this.structure[i].nodes, this.structure[i-1].nodes+1);
        }
      } else {
        this.weight_matrices[i] = this.randM(this.structure[i].nodes, this.structure[i].input_size+1);
        if (optimizer === "rprop") {
          this.prev_weight_matrices[i] = this.randM(this.structure[i].nodes, this.structure[i].input_size+1);
          this.prev_gradient_signs[i] = this.zeroM(this.structure[i].nodes, this.structure[i].input_size+1);
          this.curr_gradient[i] = this.zeroM(this.structure[i].nodes, this.structure[i].input_size+1);
          this.alpha[i] = this.zeroM(this.structure[i].nodes, this.structure[i].input_size+1);
        }
      }
      this.layers[i] = Array(this.structure[i].nodes);
      if (i !== this.structure.length-1 && this.layers[i].length === this.structure[i].nodes) {
        this.layers[i][this.layers[i].length] = 1;
      }
    }
  }

  run(x, y, epochs, learning_rate, decay_rate) {
    this.setup(this.optimizer);
    this.learning_rate = learning_rate;
    this.decay_rate = decay_rate;
    let epoch = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      this.loss = 0;
      for (let i = 0; i < x.length; i++) {
        this.feedforward(x[i].concat([1]));
        this.backpropagate(x[i].concat([1]), y[i], epoch, x.length);
        this.loss += this.difference(this.layers[this.layers.length-1], y[i])/x.length/y[i].length;
      }
      if (this.optimizer === "rprop") {
        this.rprop_update(epoch);
      }
      console.log(this.loss);
    }
  }

  feedforward(x) {
    for (let nx = 0; nx < this.layers.length; nx++) {
      if (nx === this.layers.length-1 && this.structure[nx].activation === "softmax") {
        let arr = [];
        for (let i = 0; i < this.layers[nx].length; i++) {
          let sum = 0;
          for (let j = 0; j < (this.layers.length-1?this.layers[nx-1]:x).length; j++) {
            sum += (this.layers.length-1?this.layers[nx-1][j]:x[j])*this.weight_matrices[nx][i][j];
          }
          arr.push(Math.exp(sum));
        }
        for (let i = 0; i < this.layers[nx].length; i++) {
          this.layers[nx][i] = arr[i]/arr.reduce((a, b) => a+b, 0);
        }
      }
      else if (nx) {
        for (let i = 0; i < this.layers[nx].length-(nx !== this.layers.length-1); i++) {
          let sum = 0;
          for (let j = 0; j < this.layers[nx-1].length; j++) {
            sum += this.layers[nx-1][j]*this.weight_matrices[nx][i][j];
          }
          this.layers[nx][i] = this.structure[nx].activation === "relu"?this.relu(sum):this.sigmoid(sum);
        }
      } else {
        for (let i = 0; i < this.layers[nx].length-(this.layers.length > 1 | 0); i++) {
          let sum = 0;
          for (let j = 0; j < x.length; j++) {
            sum += x[j]*this.weight_matrices[nx][i][j];
          }
          this.layers[nx][i] = this.structure[nx].activation === "relu"?this.relu(sum):this.sigmoid(sum);
        }
      }
    }
    return this.layers[this.layers.length-1];
  }

  backpropagate(x, y, epoch, N) {
    if (this.layers.length === 1) {
      for (let i = 0; i < this.layers[0].length; i++) {
        for (let j = 0; j < x.length; j++) {
          let curr_gradient = 1/(1+this.decay_rate)*1/x.length*((this.layers[0][i]-y[i])*(this.structure[0].activation === "relu"?this.relu_derivative_relu_inverse(this.layers[0][i]):this.sigmoid_derivative_sigmoid_inverse(this.layers[0][i]))*x[j]);
          if (this.optimizer === "rprop") {
            this.curr_gradient[0][i][j] += curr_gradient/N;
          } else {
            this.weight_matrices[0][i][j] -= this.learning_rate/curr_gradient;
          }
        }
      }
    } else {
      for (let nx = this.layers.length-1; nx >= 0; nx--) {
        if (nx === this.layers.length-1) {
          for (let i = 0; i < this.layers[nx].length; i++) {
            this.gradient_data[0][i] = 1/x.length*(this.layers[nx][i]-y[i])*(this.structure[nx].activation === "relu"?this.relu_derivative_relu_inverse(this.layers[nx][i]):this.sigmoid_derivative_sigmoid_inverse(this.layers[nx][i]));
            for (let j = 0; j < this.layers[nx-1].length; j++) {
              if (this.optimizer === "rprop") {
                this.curr_gradient[nx][i][j] += 1/(1+this.decay_rate)*this.gradient_data[0][i]*this.layers[nx-1][j]/N;
              } else {
                this.weight_matrices[nx][i][j] -= this.learning_rate/(1+this.decay_rate)*this.gradient_data[0][i]*this.layers[nx-1][j];
              }
            }
          }
        } else if (nx) {
          for (let j = 0; j < this.layers[nx].length-1; j++) {
            for (let k = 0; k < this.layers[nx-1].length; k++) {
              this.gradient_sums[this.layers.length-2-nx] = 0;
              for (let i = 0; i < this.layers[nx+1].length-(nx !== this.layers.length-2); i++) {
                this.gradient_sums[this.layers.length-2-nx] += this.gradient_data[this.layers.length-2-nx][i]*this.weight_matrices[nx+1][i][j];
              }
              this.gradient_data[this.layers.length-1-nx][j] = this.gradient_sums[this.layers.length-2-nx]*(this.structure[nx].activation === "relu"?this.relu_derivative_relu_inverse(this.layers[nx][j]):this.sigmoid_derivative_sigmoid_inverse(this.layers[nx][j]));
              if (this.optimizer === "rprop") {
                this.curr_gradient[nx][j][k] += 1/(1+this.decay_rate)*this.gradient_data[nx][j]*this.layers[nx-1][k]/N;
              } else {
                this.weight_matrices[nx][j][k] -= this.learning_rate/(1+this.decay_rate)*this.gradient_data[nx][j]*this.layers[nx-1][k];
              }
            }
          }
        } else {
          for (let j = 0; j < this.layers[0].length-1; j++) {
            for (let k = 0; k < x.length; k++) {
              this.gradient_sums[this.layers.length-2] = 0;
              for (let i = 0; i < this.layers[1].length-(this.layers.length !== 2); i++) {
                this.gradient_sums[this.layers.length-2] += this.gradient_data[this.layers.length-2][i]*this.weight_matrices[1][i][j];
              }
              this.gradient_data[this.layers.length-1][j] = this.gradient_sums[this.layers.length-2]*(this.structure[0].activation === "relu"?this.relu_derivative_relu_inverse(this.layers[0][j]):this.sigmoid_derivative_sigmoid_inverse(this.layers[0][j]));
              if (this.optimizer === "rprop") {
                this.curr_gradient[0][j][k] += 1/(1+this.decay_rate)*this.gradient_data[this.layers.length-1][j]*x[k]/N;
              } else {
                this.weight_matrices[0][j][k] -= this.learning_rate/(1+this.decay_rate)*this.gradient_data[this.layers.length-1][j]*x[k];
              }
            }
          }
        }
      }
    }
  }

  rprop_update(epoch) {
    for (let nx = 0; nx < this.curr_gradient.length; nx++) {
      for (let i = 0; i < this.curr_gradient[nx].length; i++) {
        for (let j = 0; j < this.curr_gradient[nx][i].length; j++) {
          if (!epoch) {
            this.alpha[nx][i][j] = this.learning_rate;
            this.weight_matrices[nx][i][j] -= this.alpha[nx][i][j]*this.curr_gradient[nx][i][j];
          } else {
            if (this.prev_gradient_signs[nx][i][j] === Math.sign(this.curr_gradient[nx][i][j])) {
              this.weight_matrices[nx][i][j] -= this.alpha[nx][i][j]*Math.sign(this.curr_gradient[nx][i][j]);
              if (this.curr_gradient[nx][i][j]) {
                this.alpha[nx][i][j] *= 1.1;
              }
            } else {
              this.alpha[nx][i][j] *= .1;
            }
          }
          this.prev_weight_matrices[nx][i][j] = this.weight_matrices[nx][i][j];
          this.prev_gradient_signs[nx][i][j] = Math.sign(this.curr_gradient[nx][i][j]);
        }
      }
    }
    for (let nx = 0; nx < this.curr_gradient.length; nx++) {
      for (let i = 0; i < this.curr_gradient[nx].length; i++) {
        for (let j = 0; j < this.curr_gradient[nx][i].length; j++) {
          this.curr_gradient[nx][i][j] = 0;
        }
      }
    }
  }

  check_nonbinary_classification_accuracy(x, y) {
    let p = 0;
    for (let nx = 0; nx < x.length; nx++) {
      this.feedforward(x[nx].concat([1]));
      p += (this.layers[this.layers.length-1].indexOf(Math.max(...this.layers[this.layers.length-1])) === y[nx].indexOf(1))/x.length;
    }
    return p;
  }

  difference(a1, a2, abs=true) {
    let sum = 0;
    for (let i = 0; i < a1.length; i++) {
      if (abs) {
        sum += Math.abs(a1[i]-a2[i]);
      } else {
        sum += a1[i]-a2[i];
      }
    }
    return sum;
  }

  randM(rows, columns) {
    let matrix = [];
    for (let i = 0; i < rows; i++) {
      let row = [];
      for (let j = 0; j < columns; j++) {
        row[j] = Math.random()*2-1;
      }
      matrix[i] = row;
    }
    return matrix;
  }

  zeroM(rows, columns) {
    let matrix = [];
    for (let i = 0; i < rows; i++) {
      let row = [];
      for (let j = 0; j < columns; j++) {
        row[j] = 0;
      }
      matrix[i] = row;
    }
    return matrix;
  }

  randV(length) {
    let vector = [];
    for (let i = 0; i < length; i++) {
      vector[i] = Math.random()*2-1;
    }
    return vector;
  }

  zeroV(length) {
    let vector = [];
    for (let i = 0; i < length; i++) {
      vector[i] = 0;
    }
    return vector;
  }

  relu(x) {return Math.max(0, x)};
  relu_derivative_relu_inverse(y) {return y !== 0};
  sigmoid(x) {return 1/(1+Math.exp(-x))};
  sigmoid_derivative_sigmoid_inverse(y) {return y*(1-y)};
}
var network = new Neural_Network(structure=[
  {"nodes": 10, "input_size": 64, "activation": "relu"},
  {"nodes": 2, "activation": "softmax"}
]);
network.setup(optimizer="rprop");
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var draw = 0;
var mouse_x = 0;
var mouse_y = 0;
var train_x = [];
var train_y = [];
function zeroM(rows, columns) {
  matrix = [];
  for (var i = 0; i < rows; i++) {
    var row = [];
    for (var j = 0; j < columns; j++) {
      row[j] = 0;
    }
    matrix[i] = row;
  }
  return matrix;
}
var img = network.zeroM(300, 300);
var centered_img;
var pixelated_img = network.zeroM(8, 8);
var x = [];
var extrema = [299, 0, 299, 0];
function center() {
  for (var i = 0; i < 300; i++) {
    for (var j = 0; j < 300; j++) {
      if (img[i][j] === 1) {
        extrema[0] = Math.min(extrema[0], j);
        break;
      }
    }
    for (var j = 299; j >= 0; j--) {
      if (img[i][j] === 1) {
        extrema[1] = Math.max(extrema[1], j);
        break;
      }
    }
  }
  for (var j = 0; j < 300; j++) {
    for (var i = 0; i < 300; i++) {
      if (img[i][j] === 1) {
        extrema[2] = Math.min(extrema[2], i);
        break;
      }
    }
    for (var i = 299; i >= 0; i--) {
      if (img[i][j] === 1) {
        extrema[3] = Math.max(extrema[3], i);
        break;
      }
    }
  }
  centered_img = network.zeroM(extrema[3]-extrema[2], extrema[1]-extrema[0]);
  for (var i = extrema[2]; i < extrema[3]; i++) {
    for (var j = extrema[0]; j < extrema[1]; j++) {
      centered_img[i-extrema[2]][j-extrema[0]] = img[i][j];
    }
  }
  var r = centered_img[0].length%8;
  for (var i = 0; i < 8-r; i++) {
    for (var j = 0; j < centered_img.length; j++) {
      if (i%2 === 0) {
        centered_img[j].push(0);
      } else {
        centered_img[j].splice(0, 0, 0);
      }
    }
  }
  r = centered_img.length%8;
  for (var i = 0; i < 8-r; i++) {
    if (i%2 === 0) {
      centered_img.push(centered_img[0].map(x => 0));
    } else {
      centered_img.splice(0, 0, centered_img[0].map(x => 0));
    }
  }
}

function pixelate() {
  for (var i = 0; i < 8; i++) {
    for (var j = 0; j < 8; j++) {
      for (var k = i*centered_img.length/8; k < (i+1)*centered_img.length/8; k++) {
        for (var l = j*centered_img[0].length/8; l < (j+1)*centered_img[0].length/8; l++) {
          pixelated_img[i][j] += centered_img[k][l]/(centered_img.length/8*centered_img[0].length/8);
        }
      }
    }
  }
}

function guess() {
  center();
  pixelate();
  for (var i = 0; i < 8; i++) {
    for (var j = 0; j < 8; j++) {
      x[8*i+j] = parseFloat(pixelated_img[i][j].toFixed(2));
    }
  }
  network.feedforward(x);
  document.querySelector("#emotion").innerText =["HAPPY", "SAD"][network.layers[network.layers.length-1].indexOf(Math.max(...network.layers[network.layers.length-1]))];
}

function train() {
  network.run(train_x, train_y, 10000, .01, 0);
}


function addsmile() {
  center();
  pixelate();
  for (var i = 0; i < 8; i++) {
    for (var j = 0; j < 8; j++) {
      x[8*i+j] = parseFloat(pixelated_img[i][j].toFixed(2));
    }
  }
  train_x.push(x);
  train_y.push([1, 0]);
  reset();
}


function addfrown() {
  center();
  pixelate();
  for (var i = 0; i < 8; i++) {
    for (var j = 0; j < 8; j++) {
      x[8*i+j] = parseFloat(pixelated_img[i][j].toFixed(2));
    }
  }
  train_x.push(x);
  train_y.push([0, 1]);
  reset();
}


function reset(){
  img = network.zeroM(300, 300);
  centered_img;
  pixelated_img = network.zeroM(8, 8);
  x = [];
  extrema = [299, 0, 299, 0];
  ctx.clearRect(0, 0, 300, 300);
}

$("#canvas").mousemove(function(e)  {
  if (draw) {
    ctx.fillRect(e.clientX-10, e.clientY-10, 20, 20);
    for (var i = 0; i <= 20; i++) {
      for (var j = 0; j <= 20; j++) {
        if (e.clientY-10+i >= 0 && e.clientY-10+i < 300 && e.clientX-10+j >= 0 && e.clientX-10+j < 300) {
          img[e.clientY-10+i][e.clientX-10+j] = 1;
        }
      }
    }
  }
})
canvas.ontouchmove = function(e)  {
  ctx.fillRect(e.touches[0].clientX-10, e.touches[0].clientY-10, 20, 20);
  for (var i = 0; i <= 20; i++) {
    for (var j = 0; j <= 20; j++) {
      if (e.touches[0].clientY-10+i >= 0 && e.touches[0].clientY-10+i < 300 && e.touches[0].clientX-10+j >= 0 && e.touches[0].clientX-10+j < 300) {
        img[parseInt(e.touches[0].clientY)-10+i][parseInt(e.touches[0].clientX)-10+j] = 1;
      }
    }
  }
}
$("#canvas").mousedown(function() {
  draw = 1;
})
$("#canvas").mouseup(function() {
  draw = 0;
})
