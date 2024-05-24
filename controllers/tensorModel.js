const tfjs = require("@tensorflow/tfjs-node");
const { Firestore } = require("@google-cloud/firestore");
const UserError = require("../exception/ClienError");
const crypto = require("crypto");

const db = new Firestore({ projectId: "submissionmlgc-ibnukenzaadin" });

function loadModel() {
    const modelUrl =
        "https://storage.googleapis.com/bucket_mlgc_ibnukenzaadinugroho/model/submissions-model/model.json";
    return new Promise(async (resolve, reject) => {
        try {
            const model = await tfjs.loadGraphModel(modelUrl);
            resolve(model);
        } catch (error) {
            reject(error);
        }
    });
}

async function getPredictHistories() {
    let querySnapshot = await db.collection("predictions").get();

    const allDocuments = [];
    querySnapshot.forEach((doc) => {
        const documentData = doc.data();
        documentData.id = doc.id; // Add the document ID for convenience
        documentData.createdAt = new Date(
            documentData.createdAt._seconds * 1000
        );
        allDocuments.push(documentData);
    });

    return allDocuments;
}

async function saveDatatoFirestore(data) {
    let predictions = db.collection("predictions");
    await predictions.doc(data.id).set(data);
}

async function predictClassification(model, imageBuffer) {
    try {
        const tensor = tfjs.node
            .decodeJpeg(imageBuffer)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        let resultPred = await model.predict(tensor).data();
        let dataRes = {
            id: crypto.randomUUID(),
            result: resultPred > 0.5 ? "Cancer" : "Non-cancer",
            suggestion:
                resultPred > 0.5
                    ? "Segera periksa ke dokter!"
                    : "Tetap jaga kesehatan!",
            createdAt: new Date(Date.now()),
        };
        saveDatatoFirestore(dataRes);
        return dataRes;
    } catch (err) {
        throw new UserError("Terjadi kesalahan dalam melakukan prediksi", 400);
    }
}

module.exports = { loadModel, predictClassification, getPredictHistories };

//ValueError: layer: Improper config format: 'className' and 'config' must set.
