import Embeddings from '../lib/embeddings.js'

// init, `id` is the name of the folder to store the vector DB in locally
const embeddings = new Embeddings({ id: 'simple-embeddings' })
await embeddings.init()

// convert text to semantic vectors and store in DB
// no-op if text already exists in the DB
// delete the folder `.data/<id>` to clear data
await embeddings.insertText(['coffee shop', 'wifi', 'hard work', 'love peace & joy, relaxation'])
await embeddings.insertText(['☕', '😄', '❤️', '😊'])

// log the text in the DB
const map = await embeddings.getTextMap()
console.log("vectors in the DB", Object.keys(map))

// Do a query, return closest matches
const results = await embeddings.search('coffee')
console.log(results.map(item => [item.item.metadata.text, item.score]))
