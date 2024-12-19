from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import os
from grouper import get_classifier, get_embedding_for_image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Detector:
    def __init__(self, yolo_model_path='./model/detect.pt'):
        self.model = YOLO(yolo_model_path)
        self.embedding_model = get_classifier("./model/fewshot-transformer-grocery.pth")

    def infer(self, image_path):
        results = self.model(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        return boxes, labels, confs

    def crop_and_get_embeddings(self, image_path, boxes):
        img = Image.open(image_path).convert('RGB')
        embeddings = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_region = img.crop((x1, y1, x2, y2))
            """
            base_filename = os.path.basename(image_path)
            filename = f"{base_filename}_cropped_{i}.jpg"
            cropped_image_path = os.path.join("./cropped", filename)
            cropped_region.save(cropped_image_path)
            """

            cropped_embedding = get_embedding_for_image(self.embedding_model, cropped_region)
            embeddings.append(cropped_embedding)
        return np.array(embeddings)

    def dynamic_num_clusters(self, embeddings):
        max_clusters = min(10, len(embeddings))
        distortions = []
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(embeddings)
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
        elbow_point = self.find_elbow_point(distortions)
        optimal_clusters = elbow_point
        if silhouette_scores[elbow_point - 2] > 0.6:
            optimal_clusters = elbow_point

        return optimal_clusters

    def find_elbow_point(self, distortions):
        diffs = np.diff(distortions)
        second_diffs = np.diff(diffs)
        elbow_point = np.argmin(second_diffs) + 2
        return elbow_point

    def cluster_embeddings(self, embeddings):
        reshaped_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        num_clusters = self.dynamic_num_clusters(reshaped_embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(reshaped_embeddings)
        
        return cluster_assignments

    def draw_boxes_with_clusters(self, image_path, boxes, cluster_assignments):
        colors = [ "red", "lightblue", "white", "yellow", "green" ]
        cluster_to_color = {}
        unique_clusters = set(cluster_assignments)
        for i, cluster_id in enumerate(unique_clusters):
            cluster_to_color[cluster_id] = colors[i % len(colors)]
        
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cluster_color = cluster_to_color[cluster_assignments[i]]
            draw.rectangle([x1, y1, x2, y2], outline=cluster_color, width=7)

        output_dir = './detected'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        img.save(output_image_path)
        return output_image_path


