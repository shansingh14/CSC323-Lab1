import base64
import random
import time


# Task 1 - Mersenne Twister 
class MT19937:
    (W, N, M, R) = (32, 624, 397, 31)
    A = 0x9908B0DF
    U, D = 11, 0xFFFFFFFF
    S, B = 7, 0x9D2C5680
    T, C = 15, 0xEFC60000
    L = 18
    F = 1812433253

    def __init__(self, seed):
        self.lower_mask = (1 << self.R) - 1
        self.upper_mask = self.D & -self.lower_mask
        self.MT = [0] * self.N
        self.idx = self.N

        self.MT[0] = seed
        for i in range(1, self.N):
            self.MT[i] = self.D & (self.F * (self.MT[i-1] ^ 
                                             (self.MT[i-1] >> (self.W-2))) + i)
    
    def extract_number(self):
        if self.idx >= self.N:
            if self.idx > self.N:
                raise Exception("No seed present")
        self.twist()

        y = self.MT[self.idx]
        y = y ^ ((y >> self.U) & self.D)
        y = y ^ ((y << self.S) & self.B)
        y = y ^ ((y << self.T) & self.C)
        y = y ^ (y >> self.L)

        self.idx += 1
        return self.D & y
    
    def twist(self):
        for i in range(self.N):
            x = (self.MT[i] & self.upper_mask) + (self.MT[(i+1) % self.N] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:
                xA = xA ^ self.A
            self.MT[i] = self.MT[(i + self.M) % self.N] ^ xA
        self.idx = 0


# now testing the MT 
def test_mt_twister32():
    pre_seed_time = int(time.time())
    time.sleep(random.randint(5, 60))  
    
    seed = int(time.time()) 
    mt = MT19937(seed)
    
    time.sleep(random.randint(5, 60)) 
    
    first_output = mt.extract_number()
    base64_output = base64.b64encode(first_output.to_bytes(4, 'big')).decode()
    
    post_seed_time = int(time.time())
    
    return pre_seed_time, seed, post_seed_time, base64_output


def oracle():
    time.sleep(random.randint(5, 60))  
    seed = int(time.time()) 
    mt = MT19937(seed) 
    time.sleep(random.randint(5, 60)) 
    first_output = mt.extract_number()
    base64_output = base64.b64encode(first_output.to_bytes(4, 'big')).decode()
    return base64_output

result = test_mt_twister32()
print(result)
print(oracle())
