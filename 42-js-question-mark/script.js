const user = {
    name: 'jeff'
}
user.write?.code() // undefined

function whatArgumentDoesThisFunctionHave(x=undefined) {
    return x || 'no argument';
}
whatArgumentDoesThisFunctionHave() // 'no argument'
whatArgumentDoesThisFunctionHave('this') // 'this'
whatArgumentDoesThisFunctionHave(0) // 'no argument' because 0 resolves falsey
whatArgumentDoesThisFunctionHave('') // 'no argument' because 0 resolves falsey

function improvedWhatArgumentDoesThisFunctionHave(x=undefined) {
    return x ?? 'no argument';
}
improvedWhatArgumentDoesThisFunctionHave() // 'no argument'
improvedWhatArgumentDoesThisFunctionHave('this') // 'this'
improvedWhatArgumentDoesThisFunctionHave(0) // 0 because 0 resolves falsey
improvedWhatArgumentDoesThisFunctionHave('') // '' because 0 resolves falsey
improvedWhatArgumentDoesThisFunctionHave(null) // '' because 0 resolves falsey
