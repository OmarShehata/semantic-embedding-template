import OpenAI from 'openai'
import { LocalIndex } from 'vectra';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import path from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url));
const api = new OpenAI({ apiKey: process.env.OPEN_API_KEY });

export default class Embeddings {
    constructor({ 
        id, 
        dataPath = path.join(__dirname, '..', `.data/`),
    }) {
        this.id = id 
        this.vectorDBIndex = new LocalIndex(path.join(dataPath, id));
    }

    async init() {
        const index = this.vectorDBIndex
        if (!await index.isIndexCreated()) {
            await index.createIndex();
        }

        if (this.useLocalModel) {
            // list of all models here?
            // you want the ones with `embeddingModel: true`
            // https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/metadata/models3.json
            this.embedderModel = await loadModel("nomic-embed-text-v1.5.f16.gguf", { verbose: true, type: 'embedding'})
        }
    }

    async embed(text) {
        const response = await api.embeddings.create({
            'model': 'text-embedding-ada-002',
            'input': [text],
        });
        return response.data.map(item => item.embedding)[0]
    }

    async insertText(textArray) {
        const existingMap = await this.getTextMap()
        const filteredText = textArray.filter(text => existingMap[text] == null)
        if (filteredText.length == 0) {
            return
        }
        const response = await api.embeddings.create({
            'model': 'text-embedding-ada-002',
            'input': filteredText,
        });
        const vectors = response.data.map(item => item.embedding)
        const items = []
        for (let i = 0; i < filteredText.length; i++) {
            const vector = vectors[i]
            const text = filteredText[i]
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