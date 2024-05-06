const arr = [1,2,3];
const [a,b,c] = arr;
const [keep, , keep2] = arr;
const [head, ...tail] = arr;

// objects
const obj = { color: 'red', size: 3, shape: 'round' };

const { color, size } = obj;
console.log(color)

const { color, ...sizeAndShape } = obj;
console.log(color) // red
console.log(sizeAndShape) // { size: 3, shape: round }

const { shape: whatShapeIsIt } = obj;
console.log(whatShapeIsIt);

const settings = { theme: 'dark' }
const { theme: 'light', fontsize: 14 } = settings;
console.log(theme); // dark

function f([a,b,c]){
    return a+b+c
}
f([1,2,3]) // 6

function g({a,b,c}) {
    return a+b+c
}
g({a:1,b:2,c:3})
