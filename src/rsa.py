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


def miller_rabin(prime_cand, itr=20):
    max_div_by_two = 0
    count = prime_cand-1
    while count % 2 == 0:
        count //= 2
        max_div_by_two += 1

    def composite_test(test_round):
        if pow(test_round, count, prime_cand) == 1:
            return False
        for i in range(max_div_by_two):
            if pow(test_round, pow(2,i) * count, prime_cand) == prime_cand-1:
                return False
        return True
    for _ in range(itr):
        round_tester = random.randrange(2, count)
        if composite_test(round_tester):
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

def run_example():
    public_key, secret_key = create(512)
    message = 4120812949124
    cipher = encrypt(message, public_key)
    decrypted_message = decrypt(cipher, secret_key)
    assert(message == decrypted_message)


if __name__ == "__main__":
    run_example()
