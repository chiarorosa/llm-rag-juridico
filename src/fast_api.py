from fastapi import FastAPI
from pydantic import BaseModel

from main import main


class User_query(BaseModel):
    query: str


app = FastAPI()


@app.post("/query")
def query(query_input: User_query):
    return main(query_input.query)
