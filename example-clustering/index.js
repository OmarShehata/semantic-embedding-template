import { kmeans } from 'ml-kmeans'
import Embeddings from '../lib/embeddings.js'

// Get already generated embeddings (see embedding examples)
const embeddings = new Embeddings({ id: 'openai-embeddings' })
await embeddings.init()
const items = await embeddings.vectorDBIndex.listItems()
const vectors = items.map(item => item.vector)

// Cluster 
const numberOfClusters = 3;
const clusteringResult = kmeans(vectors, numberOfClusters);
const clusters = getOriginalDataBuckets(clusteringResult, items)
console.log(clusters)
// result:
// [
//   [ '😄', '❤️', '😊' ],
//   [ 'coffee shop', 'wifi', 'hard work', '☕' ],
//   [ 'love peace & joy, relaxation' ]
// ]

console.log("Trying all clusters up to given number:")
determineOptimalClusters(vectors, 6)

////// Helper functions ///////
// given cluster labels like [0, 0, 1, 1, 0]
// put the original data into clusters, like 
// [ [3 things in bucket 0], [2 things in bucket 1]]
function getOriginalDataBuckets(clusteringResult, originalDataItems) {
	const clusters = clusteringResult.clusters
	const clusterMap = {}
	for (let i = 0 ; i < clusters.length; i++) {
		const label = clusters[i]
		const item = originalDataItems[i]
		if (clusterMap[label] == undefined) clusterMap[label] = []
		clusterMap[label].push(item.metadata.text)
	}

	return Object.values(clusterMap)
}


function calculateSSE(vectors, centroids, clusterLabels) {
	let sse = 0;
  
	vectors.forEach((vector, index) => {
	  const centroid = centroids[clusterLabels[index]]; // Find the corresponding centroid for the cluster
	  const distanceSquared = vector.reduce(
		(sum, value, i) => sum + Math.pow(value - centroid[i], 2),
		0
	  ); // Calculate squared Euclidean distance
	  sse += distanceSquared;
	});
  
	return sse;
  }

async function determineOptimalClusters(vectors, maxClusters) {
	const sseList = [];

	for (let k = 1; k <= maxClusters; k++) {
		const result = kmeans(vectors, k);
		const sse = calculateSSE(vectors, result.centroids, result.clusters)
		console.log({ clusterSize: k, error: sse })
		sseList.push(sse);
	}

	return sseList; // Choose the optimal k visually or programmatically
}
