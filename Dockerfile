FROM python:3.7
COPY . /pjpeg
WORKDIR /pjpeg
RUN pip install -r requirements.txt
CMD python pjpeg_classification.py --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt --val_labels val.txt --imagedir test_images