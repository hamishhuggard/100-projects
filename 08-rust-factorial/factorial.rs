fn factorial_iterative(n: u64) -> u64 {
    let mut result = 1;
    for i in 1..=n {
        result *= i;
    }
    result
}

fn factorial_recursive(n: u64) -> u64 {
    if n == 0 {
        1
    } else {
        n * factorial_recursive(n-1)
    }
}

fn main() {
    let num = 5;
    let fact_iterative = factorial_iterative(num);
    println("the factorial of {} is {}", num, fact_iterative);
    let fact_recurs = factorial_recursive(num);
    println("the factorial of {} is {}", num, fact_recurs);
}
