## Source
Tech with Tim: https://www.youtube.com/watch?v=uN7X819DUlQ&list=WL&index=2&t=235s

## Database
AstraDB: https://www.datastax.com/products/datastax-astra
--> I couldn't use this database, i had this issue:
raise APIRequestError(raw_response, payload=json_data)
astrapy.core.api.APIRequestError: {"errors":[{"message":"No \"createCollection\" command found as \"GeneralCommand\"","errorCode":"NO_COMMAND_MATCHED"}]}
--> instead i used Pinecone

### Create index in Pinecone
Pinecone:

Dimension to create index for the embbeding model we are using openai model "text-embedding-ada-002".
The dimension of the vector that this model returns is : 1536

###  Create embeddings and storing them in Pinecone
https://docs.pinecone.io/integrations/langchain
