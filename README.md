# Semantic Embedding Template

Minimal example for doing semantic embedding, search, and clustering completely offline in NodeJS. 

- [gpt4all Node.js](https://www.npmjs.com/package/gpt4all) for converting text to semantic vectors
- [Vectra](https://github.com/Stevenic/vectra/) local vector database, single file JSON. Supports fast querying.
- Clustering using basic [ml-kmeans](https://www.npmjs.com/package/ml-kmeans)

This is a sandbox for exploring language model based projects, a collection of useful snippets that I copy into other projects, which is why it's super minimal/is just node scripts.

### Setup

Run `pnpm install`. 

### Simple embedding

`example-simple-embedding/index.js`

```
pnpm simple-embedding
```

This is the simplest example. Given an array of text, you can get semantic vectors back:

```javascript
const embeddings = new Embeddings({ id: 'simple-embeddings' })
await embeddings.init()

await embeddings.insertText(['coffee shop', 'wifi', 'hard work', 'love peace & joy, relaxation'])
console.log(await embeddings.getTextMap())
```

You can do search with this, by turning the search string into a semantic vector and using vectra to find the closest vectors to it.

```javascript
const results = await embeddings.search('coffee')
console.log(results.map(item => [item.item.metadata.text, item.score]))
// [
//   [ 'coffee shop', 0.8214959697396015 ],
//   [ 'wifi', 0.711907901740376 ],
//   [ 'hard work', 0.6709908415581982 ],
//   [ 'love peace & joy, relaxation', 0.6495931802131457 ]
// ]
```

### OpenAI embedding

`example-openai-embedding/index.js`

```
pnpm openai-embedding
```

Exactly the same as the example above, but is hooked up to the OpenAI API. Requires setting the env variable `OPEN_API_KEY`. 

### Clustering

`example-clustering/index.js`

```
pnpm clustering
```

Takes vectors and clusters them using k-means. Either you tell it how many clusters to use, or run it for ~100 cluster sizes and log the error (this is the "elbow method", you keep running it as long as the error goes down, until it starts climbing again, that's the optimal).

Example cluster output, given a cluster size of 3. You can see one cluster is "happy" ? One is "coffee/work" and the unrelated concept is in its own cluster.

```javascript
// result:
// [
//   [ '😄', '❤️', '😊' ],
//   [ 'coffee shop', 'wifi', 'hard work', '☕' ],
//   [ 'love peace & joy, relaxation' ]
// ]
```

### Appendix

- running various models directly in JS
    https://github.com/xenova/transformers.js?tab=readme-ov-file#installation
- Open image models
    - https://github.com/vikhyat/moondream
    - can convert image to text, and then semantically search that, in order to implement "CTRL+F" for images
- would be neat to add an example for how to "reverse" embeddings. Like Linus's notebook example? 