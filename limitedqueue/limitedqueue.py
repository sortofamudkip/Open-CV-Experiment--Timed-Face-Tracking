"""
A queue of size N. If the queue is at max size, pushing an element removes the least recent element pushed in.
"""
from collections import deque

class LimitedQueue:
	def __init__(self, size):
		self.size = size
		self.queue = deque()

	def push(self, element):
		self.queue.append(element)
		if len(self.queue) > self.size:
			self.queue.popleft()
			assert len(self.queue) == self.size

	def is_full(self):
		return len(self.queue) == self.size

	def and_all(self):
		if len(self.queue) == 0: return set()
		result = self.queue[0]
		for i in range(1, len(self.queue)):
			result = result & self.queue[i]
			if result == set(): break
		return result


	def __len__(self):
		return len(self.queue)

	def __str__(self):
		return str(self.queue)

if __name__ == "__main__":
	Q = LimitedQueue(5)
	Q.push({1, 2, 3})
	Q.push({2, 3, 4})
	Q.push({2, 3, 4})
	Q.push({2, 3, 4})
	Q.push({2, 3, 4})
	Q.push({2, 3, 4})
	Q.push({2, 3, 5})
	print(Q.and_all())
