# Urdu Speech Recognition

## Description

This project is basically the Urdu Voice Recognition by using speech recognition library in python. This is just the backend which is based on flask caintaing three end points. This backend is host on AWS serverless lambda service and protected by AWS WAF and AWS Sheild service. I also used AWS S3 service to store my Audio files and AWS DynamoDB to save my data.  

Here is the base URL:
```
https://zb3aeahdg2.execute-api.me-south-1.amazonaws.com/dev
```

### End Points

> /
- This is just root end point which give you the structure of response and you can validate that the server is running. This is **GET** method.

> /data
- This endpoint give the list of all the data (audio file url and prediction) which is processed by the application from the day its deploed. This is **GET** method.

> /predict
- This is the main end point in which you send your audio file in preferably **.wav** format in a body (multipart) then the app save that file in S3 bucket then apply speech recognition and then save audio url and prediction to the dynamoDB. This is **POST** method.

I have also added postman collection which you can import on your postman so you can get the examples of request and response.
