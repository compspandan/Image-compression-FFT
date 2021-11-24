import random


def pow_mod(a, b, n):
    d = 1
    while b > 0:
        if b & 1:
            d = ((d % n) * (a % n)) % n
        a = ((a % n) * (a % n)) % n
        b >>= 1
    return d


def generate_random_number(number_of_bits):
    return random.randrange((1 << (number_of_bits - 1)) + 1, (1 << number_of_bits) - 1)


def sieve_of_erathosthenes(size):
    is_prime = [True] * (size + 1)
    is_prime[0] = is_prime[1] = False
    primes = list()
    for i in range(2, size + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, size + 1, i):
                is_prime[j] = False
    return primes


def get_low_prime(number_of_bits):
    primes = sieve_of_erathosthenes(1000)
    while True:
        is_coprime = True
        prime_candidate = generate_random_number(number_of_bits)
        for prime in primes:
            if prime_candidate % prime == 0:
                is_coprime = False
                break
        if is_coprime:
            return prime_candidate


def miller_rabin(mrc, iterations=20):
    maxDivisionsByTwo = 0
    ec = mrc-1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert(2**maxDivisionsByTwo * ec == mrc-1)

    def trialComposite(round_tester):
        if pow(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if pow(round_tester, 2**i * ec, mrc) == mrc-1:
                return False
        return True
    for i in range(iterations):
        round_tester = random.randrange(2, mrc)
        if trialComposite(round_tester):
            return False
    return True


def select_random_prime(number_of_bits):
    while True:
        prime_candidate = get_low_prime(number_of_bits)
        if miller_rabin(prime_candidate):
            return prime_candidate


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def modular_mulitplicative_inverse(a, m):
    m0 = m
    y = 0
    x = 1

    if (m == 1):
        return 0

    while (a > 1):
        q = a // m

        t = m

        m = a % m
        a = t
        t = y

        y = x - q * y
        x = t
    if (x < 0):
        x = x + m0

    return x


def create(number_of_bits):
    p = select_random_prime(number_of_bits)
    q = select_random_prime(number_of_bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 3

    while e < phi:
        if gcd(e, phi) == 1:
            break
        e += 2

    if e >= phi:
        raise Exception("FAILED")

    d = modular_mulitplicative_inverse(e, phi)
    public_key = (e, n)
    secret_key = (d, n)

    return public_key, secret_key


def encrypt(message, public_key):
    e, n = public_key
    return pow_mod(message, e, n)


def decrypt(cipher, secret_key):
    d, n = secret_key
    return pow_mod(cipher, d, n)


public_key, secret_key = create(512)
message = 4120812949124
cipher = encrypt(message, public_key)
decrypted_message = decrypt(cipher, secret_key)
assert(message == decrypted_message)
