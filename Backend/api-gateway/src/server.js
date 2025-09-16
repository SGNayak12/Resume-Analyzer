import express from "express";
import cors from "cors";
import bodyParser from "body-parser";

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

app.get("/", (req, res) => {
  res.send("API Gateway Running 🚀");
});

app.get("/auth/status", (req, res) => {
  res.json({ status: "Auth service would respond here" });
});

app.listen(PORT, () => console.log(`API Gateway listening on ${PORT}`));
