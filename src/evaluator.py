import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple, List, Dict
import json


class EmotionTagger:

    def __init__(self, thresholds=None):
        if thresholds is None:
            # Simplified thresholds, as valence/arousal now use 0 as the division line
            thresholds = {
                "neutral_radius": 0.1  # Using the value from XUDRSho1rDqy which seems reasonable for a neutral zone.
            }
        self.thresholds = thresholds

        # 5 emotions (4 quadranty + neutral)
        self.emotion_labels = ["NEUTRAL", "HAPPY", "CALM", "ANGRY", "SAD"]

        # Mapping dla każdej emocji
        self.emotion_colors = {
            "NEUTRAL": "#808080",  # Gray
            "HAPPY": "#FFD700",  # Gold
            "CALM": "#87CEEB",  # Sky blue
            "ANGRY": "#FF6347",  # Tomato red
            "SAD": "#4169E1",  # Royal blue
        }

    def tag_emotion(self, valence: float, arousal: float) -> str:
        neutral_r = self.thresholds["neutral_radius"]

        # Neutral zone around (0,0)
        dist = np.sqrt(valence**2 + arousal**2)
        if dist < neutral_r:
            return "NEUTRAL"

        # Quadrant classification using raw valence and arousal
        if valence >= 0 and arousal >= 0:
            return "HAPPY"  # Positive + Active
        elif valence >= 0 and arousal < 0:
            return "CALM"  # Positive + Passive
        elif valence < 0 and arousal >= 0:
            return "ANGRY"  # Negative + Active
        else:  # valence < 0 and arousal < 0
            return "SAD"  # Negative + Passive

    def tag_batch(self, valences: np.ndarray, arousals: np.ndarray) -> np.ndarray:
        emotions = np.array([self.tag_emotion(v, a) for v, a in zip(valences, arousals)])
        return emotions

    def emotion_to_label_id(self, emotion: str) -> int:
        """Convert emotion string → numeric ID."""
        return self.emotion_labels.index(emotion)

    def label_id_to_emotion(self, label_id: int) -> str:
        """Convert numeric ID → emotion string."""
        return self.emotion_labels[label_id]


class EmotionEvaluator:

    def __init__(self, tagger: EmotionTagger = None):
        self.tagger = tagger if tagger else EmotionTagger()

    def evaluate(
        self, true_valences: np.ndarray, true_arousals: np.ndarray, pred_valences: np.ndarray, pred_arousals: np.ndarray
    ) -> Dict:

        # Tag emotions
        true_emotions = self.tagger.tag_batch(true_valences, true_arousals)
        pred_emotions = self.tagger.tag_batch(pred_valences, pred_arousals)

        # Convert to numeric IDs dla metryk
        true_ids = np.array([self.tagger.emotion_to_label_id(e) for e in true_emotions])
        pred_ids = np.array([self.tagger.emotion_to_label_id(e) for e in pred_emotions])

        # Metrics
        accuracy = accuracy_score(true_ids, pred_ids)

        # Per-emotion accuracy
        per_emotion_acc = {}
        for emotion in self.tagger.emotion_labels:
            emotion_mask = true_emotions == emotion
            if emotion_mask.sum() > 0:
                emotion_acc = accuracy_score(true_ids[emotion_mask], pred_ids[emotion_mask])
                per_emotion_acc[emotion] = emotion_acc

        # Confusion matrix
        cm = confusion_matrix(true_ids, pred_ids, labels=range(len(self.tagger.emotion_labels)))

        # Classification report
        class_report = classification_report(
            true_ids,
            pred_ids,
            target_names=self.tagger.emotion_labels,
            labels=range(len(self.tagger.emotion_labels)),  # <--- ADDED THIS LINE
            output_dict=True,
            zero_division=0,
        )

        # Distribution stats
        dist_true = {emotion: (true_emotions == emotion).sum() for emotion in self.tagger.emotion_labels}
        dist_pred = {emotion: (pred_emotions == emotion).sum() for emotion in self.tagger.emotion_labels}

        results = {
            "overall_accuracy": float(accuracy),
            "per_emotion_accuracy": {k: float(v) for k, v in per_emotion_acc.items()},
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "true_distribution": dist_true,
            "pred_distribution": dist_pred,
            "true_emotions": true_emotions.tolist(),
            "pred_emotions": pred_emotions.tolist(),
            "true_ids": true_ids.tolist(),
            "pred_ids": pred_ids.tolist(),
        }

        return results

    def plot_confusion_matrix(self, results: Dict, save_path="emotion_confusion_matrix.png"):
        """Plot confusion matrix."""
        cm = np.array(results["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.tagger.emotion_labels,
            yticklabels=self.tagger.emotion_labels,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted Emotion")
        ax.set_ylabel("True Emotion")
        ax.set_title(f"Emotion Classification Confusion Matrix\n" f"Accuracy: {results['overall_accuracy']:.1%}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved confusion matrix: {save_path}")
        plt.close()

    def plot_distributions(self, results: Dict, save_path="emotion_distributions.png"):
        """Plot true vs predicted emotion distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        emotions = self.tagger.emotion_labels
        true_counts = [results["true_distribution"][e] for e in emotions]
        pred_counts = [results["pred_distribution"][e] for e in emotions]

        x = np.arange(len(emotions))
        width = 0.35

        # True distribution
        axes[0].bar(x - width / 2, true_counts, width, label="True", alpha=0.8)
        axes[0].bar(x + width / 2, pred_counts, width, label="Predicted", alpha=0.8)
        axes[0].set_xlabel("Emotion")
        axes[0].set_ylabel("Count")
        axes[0].set_title("True vs Predicted Emotion Distribution")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(emotions)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Per-emotion accuracy
        per_emotion_acc = results["per_emotion_accuracy"]
        accs = [per_emotion_acc.get(e, 0) for e in emotions]
        colors = [self.tagger.emotion_colors.get(e, "#808080") for e in emotions]

        axes[1].bar(emotions, accs, color=colors, alpha=0.7)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Per-Emotion Classification Accuracy")
        axes[1].set_ylim([0, 1])
        axes[1].grid(alpha=0.3, axis="y")

        # Add percentage labels on bars
        for i, (emotion, acc) in enumerate(zip(emotions, accs)):
            axes[1].text(i, acc + 0.02, f"{acc:.0%}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved distributions: {save_path}")
        plt.close()

    def plot_valence_arousal_scatter(
        self,
        true_valences: np.ndarray,
        true_arousals: np.ndarray,
        pred_valences: np.ndarray,
        pred_arousals: np.ndarray,
        save_path="emotion_scatter.png",
    ):

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # True emotions scatter
        ax = axes[0]
        true_emotions = self.tagger.tag_batch(true_valences, true_arousals)

        for emotion in self.tagger.emotion_labels:
            mask = true_emotions == emotion
            ax.scatter(
                true_valences[mask],
                true_arousals[mask],
                label=emotion,
                color=self.tagger.emotion_colors[emotion],
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        # Linie podziału dla zakresu [-1,1]
        ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Neutral zone wokół środka (0,0)
        from matplotlib.patches import Circle

        neutral_r = self.tagger.thresholds["neutral_radius"]
        circle = Circle((0.0, 0.0), neutral_r, fill=False, linestyle="--", color="gray", alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("True Emotions (Valence-Arousal Space)")
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.legend()
        ax.grid(alpha=0.3)

        # Predicted emotions scatter
        ax = axes[1]
        pred_emotions = self.tagger.tag_batch(pred_valences, pred_arousals)

        for emotion in self.tagger.emotion_labels:
            mask = pred_emotions == emotion
            ax.scatter(
                pred_valences[mask],
                pred_arousals[mask],
                label=emotion,
                color=self.tagger.emotion_colors[emotion],
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        # Linie podziału dla zakresu [-1,1]
        ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Neutral zone wokół środka (0,0)
        circle = Circle((0.0, 0.0), neutral_r, fill=False, linestyle="--", color="gray", alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("Predicted Emotions (Valence-Arousal Space)")
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved scatter plot: {save_path}")
        plt.close()
