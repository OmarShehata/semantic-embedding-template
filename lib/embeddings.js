import { LocalIndex } from 'vectra';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import path from 'path'
import { loadModel, createEmbedding } from 'gpt4all'

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 // local usage:
 
 const embeddings = new Embeddings({ id: 'unique_id_for_local_db' })
 await embeddings.init()

 // insert text if it's not already in the DB
 embeddings.insertText(["one", "two", "banana"]))

 // find the closest items to the query string
 const items = await embeddings.search("search string") 
 */
export default class Embeddings {
    constructor({ 
        id, 
        dataPath = path.join(__dirname, '..', `.data/`)
    }) {
        this.id = id 
        this.vectorDBIndex = new LocalIndex(path.join(dataPath, id));
        this.embedderModel = null
    }

    async init() {
        const index = this.vectorDBIndex
        if (!await index.isIndexCreated()) {
            await index.createIndex();
        }

        // list of all models here?
        // you want the ones with `embeddingModel: true`
        // https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/metadata/models3.json
        this.embedderModel = await loadModel("nomic-embed-text-v1.5.f16.gguf", { type: 'embedding'})
    }

    async embed(text) {
        const { embeddings } = await createEmbedding(this.embedderModel, text)
        return Array.from(embeddings)
    }

    async insertText(textArray) {
        const existingMap = await this.getTextMap()

        const items = []
        for (let text of textArray) {
            if (existingMap[text]) continue 

            const vector = await this.embed(text)
            items.push({ vector, text })
        }

        const index = this.vectorDBIndex
        await index.beginUpdate();
        for (let item of items) {
            await index.insertItem({
                vector: item.vector,
                metadata: { text: item.text }
            });
        }
        await index.endUpdate();
    }

    async search(text, max = 100) {
        const index = this.vectorDBIndex
        const searchVector = await this.embed(text)
        const results = await index.queryItems(searchVector, max);
        return results
    }

    // Given text, find the item & its associated vector
    async findByText(text) {
        const index = this.vectorDBIndex
        return await index.listItemsByMetadata({ text: { $eq: text } })
    }
    // map of embedded text -> vector
    async getTextMap() {
        const allItems = await this.vectorDBIndex.listItems()
        const itemMap = {}
        for (let item of allItems) {
            const key = item.metadata.text
            itemMap[key] = item
        }

        return itemMap
    }
}