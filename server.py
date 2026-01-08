from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Query
from client.rq_client import queue
from queues.worker import process_query

app = FastAPI()

@app.get('/')
def root():
    return {"Server":'Server is up and running'}

@app.post('/chat')
def chat(
    query = Query(...,description="The Chat Query of User")
):
    job= queue.enqueue(process_query,query)
    return {"status":"queued", "job_id":job.id}
@app.get("/job-status")
def getResult(
        job_id: str = Query(...,description="Job ID")
):
    job = queue.fetch_job(job_id=job_id)
    result = job.return_value()
    return{"result": result}