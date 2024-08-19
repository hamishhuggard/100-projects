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

// functions
function f([a,b,c]){
    return a+b+c
}
f([1,2,3]) // 6

function g({a,b,c}) {
    return a+b+c
}
g({a:1,b:2,c:3})

// nexted objects
const Parent = {
    child: {
        color: 'red'
    }
};
const { parent: { child } } = Parent;

// loops
const users = [
    { id: 1 },
    { id: 2 },
    { id: 3 }
];
for (let { id } of users) {
    console.log(id);
}

// variable swapping
let [a,b] = [1,2];
[b,a] = [a,b];

// regex
let re = /\w+\s/g;
let str = 'fe fi fo';
const [fe, fi, fo] = str.match(re);

// dynamic properties
const rando = randomKey();
const obj = {
    [rando]: 23
}
const { [rando]: myKey } = obj;
