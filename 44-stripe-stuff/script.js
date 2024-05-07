{
    "id": "pi_...",
    "amount": 2300,
    "status": "requires_payment_method",
    "client_secret": "..."
}

const result = await stripe.confirmCardPayment (
    paymentIntent.client_secret, {
        payment_method: { card }
    }
)
