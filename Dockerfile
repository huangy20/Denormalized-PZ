FROM python:3.13

ADD . .

RUN pip3 install numpy scipy matplotlib

CMD ["python3", "all_in_one_script.py"]
