import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import os
import pandas as pd

BASE_URL = "https://www.luckyscent.com"

PERFUME_URL = BASE_URL + "/category/1/fragrances?sort=new&sortdir=1&page="

if not os.path.exists('perfume_urls.txt'):
	master_list =[]
	master_text_list = []
	for i in range(1, 70):
		print("Extracting Data From ", PERFUME_URL+str(i))
		page = requests.get(PERFUME_URL+str(i))

		soup = BeautifulSoup(page.text, 'html.parser')

		perfs = soup.find_all("div", class_="row search-results row-items-4 ls-grid")
		for perf in perfs:
			perfume_list = perf.find_all("div", class_="search-item item col-xs-6 col-sm-4 col-md-3")

			print(len(perfume_list))
			for perfume in perfume_list:
				element_url = perfume.a['href']
				print(element_url)
				text = perfume.find("div", class_="search-item-content").text
				text_list = text.split('\n')
				text_list = [txt.strip() for txt in text_list]
				text_list = [txt for txt in text_list if txt != '']
				# print(text_list)
				master_list.append(element_url)
				master_text_list.append(text_list)

	with open('perfume_urls.txt', 'w') as fhead:
		print('Writing perfume_urls.txt...')
		for url in master_list:
			fhead.write(url+"\n")
	with open('perfume_text.txt', 'w') as fhead:
		print('Writing perfume_text.txt...')
		for text in master_text_list:
			fhead.write(' ### '.join(text)+"\n")

else:
	print('perfume_urls.txt Found...')
	print('Reading perfume_urls.txt..')

	with open('perfume_urls.txt', 'r') as fhead:
		master_list = fhead.readlines()

	with open('perfume_text.txt', 'r') as fhead:
		master_text_list = fhead.readlines()

data = []
for url, details in zip(master_list, master_text_list):
	perfume_details = details.split(' ### ')
	brand = perfume_details[0]
	if len(perfume_details) == 3:
		perfume_name = perfume_details[1] + ' ' + perfume_details[2]
	else:
		perfume_name = perfume_details[1]
	perfume_name = perfume_name.strip()

	des = 0
	notes = 0
	perf_description = ''
	perf_notes = ''
	url = url.strip()
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	perf_block = soup.find_all("div", class_="container-fluid product-container")
	for block in perf_block:
		perf_img_block = block.find("div", class_="product-image product-image-container")
		img_url = perf_img_block.img['src']
		print(img_url)
		perfume_text = block.find("div", class_="product-details").text
		perf_text_list = perfume_text.split('\n')
		perf_text_list = [txt.strip() for txt in perf_text_list]
		for txt in perf_text_list:
			if txt != '':
				if 'Scoop' in txt:
					des = 1
				if 'Fragrance Notes' in txt:
					des = 0
					notes = 1

				if des == 1 and 'Scoop' not in txt:
					perf_description = perf_description + ' ' + txt
				if notes == 1 and 'Fragrance Notes' not in txt:
					perf_notes = perf_notes + ' ' + txt
		print("Name ", perfume_name)
		print("Brand ", brand)
		print("Description ", perf_description)
		print("Notes ", perf_notes)
		data.append([perfume_name, brand, perf_description, perf_notes, img_url])
	print("**************")

	df = pd.DataFrame(data, columns=['Name', 'Brand', 'Description', 'Notes', 'Image URL'])
	df.to_csv('final_perfume_data.csv', index=False, encoding='utf-8')