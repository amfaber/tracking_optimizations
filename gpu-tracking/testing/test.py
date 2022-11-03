from email.mime import base
from bs4 import BeautifulSoup
import requests
import requests.exceptions
from urllib.parse import urlsplit
from collections import deque
import re
import sys

# a queue of urls to be crawled

new_urls = deque(['https://www.kriminalforsorgen.dk/'])
processed_urls = set()

# with open("currently.txt") as file:
# 	new_urls = deque(file.read().split("\n"))
# with open("processed.txt") as file:
# 	processed_urls = set(file.read().split("\n"))

# a set of urls that we have already crawled

# a set of crawled emails
emails = set()
i = 0
with open("mails.txt", "w") as file:
	# process urls one by one until we exhaust the queue
	while len(new_urls):

		# print(i)
		# i += 1
		# move next url from the queue to the set of processed urls
		url = new_urls.popleft()
		processed_urls.add(url)

		# extract base url to resolve relative links
		parts = urlsplit(url)
		base_url = "{0.scheme}://{0.netloc}".format(parts)
		# print(base_url)
		path = url[:url.rfind('/')+1] if '/' in parts.path else url
		# print(path)
		# sys.exit()

		# get url's content
		print("Processing %s" % url)
		try:
			response = requests.get(url)
		except (requests.exceptions.MissingSchema, requests.exceptions.ConnectionError):
			# ignore pages with errors
			continue

		# extract all email addresses and add them into the resulting set
		new_emails = set(re.findall(r"[\w\d\.\-+_]+@[\w\d\.\-+_]+\.[\w]+", response.text, re.I))
		for mail in new_emails:
			if mail not in emails:
				file.write(path)
				file.write("\t")
				file.write(mail)
				file.write("\n")
		file.flush()
		emails.update(new_emails)

		# create a beutiful soup for the html document
		try:
			soup = BeautifulSoup(response.text)
			for anchor in soup.find_all("a"):
				# extract link url from the anchor
				link = anchor.attrs["href"] if "href" in anchor.attrs else ''
				# resolve relative links
				# print(f"This is a link: {link}")
				# sys.exit()
				if link.startswith('/'):
					# print(f"link in here {link}")
					link = base_url + link
				elif not link.startswith('http'):
					link = path + link
				link = link.replace("#content", "")
				# add the new url to the queue if it was not enqueued nor processed yet
				split = link.split(base_url)
				do_we_care_about_the_link = (
					not link in new_urls and \
					not link in processed_urls and \
					len(split) == 2 and \
					split[0] == "" and \
					not "." in split[1]
				)
				if do_we_care_about_the_link:
					new_urls.append(link)
			with open("processed.txt", "w") as file1, open("currently.txt", "w") as file2:
				file1.write("\n".join(processed_urls))
				file2.write("\n".join(new_urls))
			
		except AssertionError:
			pass
			# find and process all the anchors in the document
			
		
		
