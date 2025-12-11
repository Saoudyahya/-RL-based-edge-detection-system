"""
Streamlit App for RL-Based Edge Detection Pipeline Optimizer
============================================================
"""

import streamlit as st
import numpy as np
import cv2
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import io
from PIL import Image
import tempfile
import os


# ============================================================================
# OPERATORS
# ============================================================================

class PreprocessingOperators:
    @staticmethod
    def gaussian_blur(image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def median_filter(image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def histogram_equalization(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(image)

    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def normalization(image):
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def no_preprocessing(image):
        return image.copy()


class ProcessingOperators:
    @staticmethod
    def sobel(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def prewitt(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        grad_x = cv2.filter2D(image.astype(float), -1, kernel_x)
        grad_y = cv2.filter2D(image.astype(float), -1, kernel_y)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def scharr(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def laplacian(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(laplacian, 50, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def canny(image, threshold1=50, threshold2=150):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(image, threshold1, threshold2)

    @staticmethod
    def canny_adaptive(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = np.median(image)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        return cv2.Canny(image, lower, upper)


class PostprocessingOperators:
    @staticmethod
    def dilation(image, kernel_size=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def erosion(image, kernel_size=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def opening(image, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def closing(image, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def adaptive_threshold(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def non_maximum_suppression(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return gradient

    @staticmethod
    def no_postprocessing(image):
        return image.copy()


PREPROCESSING_OPS = {
    0: ('gaussian_blur', PreprocessingOperators.gaussian_blur),
    1: ('median_filter', PreprocessingOperators.median_filter),
    2: ('histogram_equalization', PreprocessingOperators.histogram_equalization),
    3: ('bilateral_filter', PreprocessingOperators.bilateral_filter),
    4: ('normalization', PreprocessingOperators.normalization),
    5: ('no_preprocessing', PreprocessingOperators.no_preprocessing),
}

PROCESSING_OPS = {
    0: ('sobel', ProcessingOperators.sobel),
    1: ('prewitt', ProcessingOperators.prewitt),
    2: ('scharr', ProcessingOperators.scharr),
    3: ('laplacian', ProcessingOperators.laplacian),
    4: ('canny', ProcessingOperators.canny),
    5: ('canny_adaptive', ProcessingOperators.canny_adaptive),
}

POSTPROCESSING_OPS = {
    0: ('dilation', PostprocessingOperators.dilation),
    1: ('erosion', PostprocessingOperators.erosion),
    2: ('opening', PostprocessingOperators.opening),
    3: ('closing', PostprocessingOperators.closing),
    4: ('adaptive_threshold', PostprocessingOperators.adaptive_threshold),
    5: ('non_maximum_suppression', PostprocessingOperators.non_maximum_suppression),
    6: ('no_postprocessing', PostprocessingOperators.no_postprocessing),
}


# ============================================================================
# METRICS
# ============================================================================

def intersection_over_union(pred, gt):
    pred = (pred > 127).astype(np.uint8)
    gt = (gt > 127).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0.0


def dice_coefficient(pred, gt):
    pred = (pred > 127).astype(np.uint8)
    gt = (gt > 127).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0


def f1_score(pred, gt):
    pred = (pred > 127).astype(np.uint8)
    gt = (gt > 127).astype(np.uint8)
    true_positive = np.logical_and(pred == 1, gt == 1).sum()
    false_positive = np.logical_and(pred == 1, gt == 0).sum()
    false_negative = np.logical_and(pred == 0, gt == 1).sum()
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def ssim_score(pred, gt):
    if pred.shape != gt.shape:
        return 0.0
    if len(pred.shape) == 3:
        pred = pred[:, :, 0]
    if len(gt.shape) == 3:
        gt = gt[:, :, 0]
    return ssim(pred, gt, data_range=255)


def compute_reward(pred, gt, metric='iou'):
    if metric == 'iou':
        return intersection_over_union(pred, gt)
    elif metric == 'dice':
        return dice_coefficient(pred, gt)
    elif metric == 'f1':
        return f1_score(pred, gt)
    elif metric == 'ssim':
        return (ssim_score(pred, gt) + 1) / 2
    else:
        return intersection_over_union(pred, gt)


# ============================================================================
# RL FUNCTION
# ============================================================================

def find_best_pipeline_streamlit(image, ground_truth, episodes, reward_metric, progress_bar, status_text):
    """Find the optimal edge detection pipeline using RL with Streamlit progress tracking"""

    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(ground_truth.shape) == 3:
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    if image.shape != ground_truth.shape:
        ground_truth = cv2.resize(ground_truth, (image.shape[1], image.shape[0]))

    # Q-Learning setup
    q_table = defaultdict(lambda: defaultdict(float))
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    learning_rate = 0.1
    discount_factor = 0.95

    best_reward = -float('inf')
    best_pipeline = None
    best_result = None
    history = []

    # RL exploration
    for episode in range(episodes):
        # Reset
        current_img = image.copy()
        pipeline = {}

        # Phase 1: Preprocessing
        state = 0
        if np.random.random() < epsilon:
            action = np.random.randint(0, len(PREPROCESSING_OPS))
        else:
            q_values = [q_table[state][a] for a in range(len(PREPROCESSING_OPS))]
            action = np.argmax(q_values)

        op_name, op_func = PREPROCESSING_OPS[action]
        current_img = op_func(current_img)
        pipeline['preprocessing'] = (action, op_name)

        # Phase 2: Processing (Edge Detection)
        state = 1
        if np.random.random() < epsilon:
            action = np.random.randint(0, len(PROCESSING_OPS))
        else:
            q_values = [q_table[state][a] for a in range(len(PROCESSING_OPS))]
            action = np.argmax(q_values)

        op_name, op_func = PROCESSING_OPS[action]
        current_img = op_func(current_img)
        pipeline['processing'] = (action, op_name)

        # Phase 3: Postprocessing
        state = 2
        if np.random.random() < epsilon:
            action = np.random.randint(0, len(POSTPROCESSING_OPS))
        else:
            q_values = [q_table[state][a] for a in range(len(POSTPROCESSING_OPS))]
            action = np.argmax(q_values)

        op_name, op_func = POSTPROCESSING_OPS[action]
        current_img = op_func(current_img)
        pipeline['postprocessing'] = (action, op_name)

        # Compute reward
        reward = compute_reward(current_img, ground_truth, metric=reward_metric)

        # Update Q-values
        for phase_idx, phase_name in enumerate(['preprocessing', 'processing', 'postprocessing']):
            action_taken = pipeline[phase_name][0]
            q_table[phase_idx][action_taken] = q_table[phase_idx][action_taken] + \
                                               learning_rate * (reward - q_table[phase_idx][action_taken])

        # Track best
        if reward > best_reward:
            best_reward = reward
            best_pipeline = pipeline.copy()
            best_result = current_img.copy()

        history.append(reward)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update progress
        if (episode + 1) % 10 == 0:
            progress_bar.progress((episode + 1) / episodes)
            avg_reward = np.mean(history[-100:]) if len(history) >= 100 else np.mean(history)
            status_text.text(
                f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.4f} | Best: {best_reward:.4f}")

    # Compute final metrics
    metrics = {
        'iou': intersection_over_union(best_result, ground_truth),
        'dice': dice_coefficient(best_result, ground_truth),
        'f1': f1_score(best_result, ground_truth),
        'ssim': ssim_score(best_result, ground_truth),
    }

    return {
        'pipeline': best_pipeline,
        'result': best_result,
        'metrics': metrics,
        'reward_history': history,
        'best_reward': best_reward
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plot(original, ground_truth, predicted, metrics, pipeline, history):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ground_truth, cmap='gray')
    ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(predicted, cmap='gray')
    ax3.set_title('RL Agent Result', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Comparison overlay
    gt_binary = (ground_truth > 127).astype(np.uint8)
    pred_binary = (predicted > 127).astype(np.uint8)
    overlay = np.zeros((*original.shape, 3), dtype=np.uint8)
    overlay[..., 1] = np.logical_and(pred_binary, gt_binary).astype(np.uint8) * 255  # Green
    overlay[..., 0] = np.logical_and(pred_binary, ~gt_binary.astype(bool)).astype(np.uint8) * 255  # Red
    overlay[..., 2] = np.logical_and(~pred_binary.astype(bool), gt_binary).astype(np.uint8) * 255  # Blue

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlay)
    ax4.set_title('Comparison\n🟢 Correct  🔴 False+  🔵 False-', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Row 2: Learning curve and metrics
    ax5 = fig.add_subplot(gs[1, :3])
    episodes = list(range(len(history)))
    ax5.plot(episodes, history, alpha=0.6, color='blue', label='Episode Reward')

    # Moving average
    window = 50
    if len(history) >= window:
        moving_avg = np.convolve(history, np.ones(window) / window, mode='valid')
        ax5.plot(episodes[window - 1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Avg')

    ax5.set_xlabel('Episode', fontsize=12)
    ax5.set_ylabel('Reward', fontsize=12)
    ax5.set_title('RL Learning Progress', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Pipeline and metrics info
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')

    info_text = "OPTIMAL PIPELINE:\n" + "=" * 30 + "\n\n"
    for phase, (idx, name) in pipeline.items():
        info_text += f"{phase.upper()}:\n  → {name}\n\n"

    info_text += "=" * 30 + "\n"
    info_text += "METRICS:\n" + "=" * 30 + "\n\n"
    info_text += f"IoU:   {metrics['iou']:.4f}\n"
    info_text += f"Dice:  {metrics['dice']:.4f}\n"
    info_text += f"F1:    {metrics['f1']:.4f}\n"
    info_text += f"SSIM:  {metrics['ssim']:.4f}\n"

    ax6.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('RL-Based Edge Detection Results', fontsize=16, fontweight='bold', y=0.98)

    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="RL Edge Detection Optimizer", layout="wide", page_icon="🔍")

    st.title("🔍 RL-Based Edge Detection Pipeline Optimizer")
    st.markdown("""
    This app uses **Reinforcement Learning** to find the optimal combination of image processing operations 
    for edge detection. Upload your original image and ground truth, and let the RL agent explore 
    the best pipeline for you!
    """)

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    episodes = st.sidebar.slider(
        "Number of Episodes",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="More episodes = better optimization but slower processing"
    )

    reward_metric = st.sidebar.selectbox(
        "Reward Metric",
        options=['iou', 'dice', 'f1', 'ssim'],
        index=0,
        help="Metric to optimize during training"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 How it works
    1. **Upload** your original image and ground truth
    2. **Configure** RL parameters
    3. **Run** the optimization
    4. **Download** the best result

    The RL agent explores combinations of:
    - **Preprocessing** (6 operations)
    - **Edge Detection** (6 operators)
    - **Postprocessing** (7 operations)

    Total search space: **252 combinations**
    """)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Original Image")
        original_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            key="original"
        )

        if original_file:
            original_image = Image.open(original_file)
            st.image(original_image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("📤 Upload Ground Truth")
        gt_file = st.file_uploader(
            "Choose ground truth edge map",
            type=['png', 'jpg', 'jpeg'],
            key="ground_truth"
        )

        if gt_file:
            gt_image = Image.open(gt_file)
            st.image(gt_image, caption="Ground Truth", use_container_width=True)

    # Run button
    st.markdown("---")

    if st.button("🚀 Run RL Optimization", type="primary", use_container_width=True):
        if original_file is None or gt_file is None:
            st.error("⚠️ Please upload both original image and ground truth!")
        else:
            # Convert to numpy arrays
            original_np = np.array(original_image)
            gt_np = np.array(gt_image)

            # Convert RGB to BGR for OpenCV
            if len(original_np.shape) == 3:
                original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
            if len(gt_np.shape) == 3:
                gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)

            # Progress tracking
            st.subheader("🎯 Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Run RL optimization
            with st.spinner("Training RL agent..."):
                result = find_best_pipeline_streamlit(
                    original_np,
                    gt_np,
                    episodes,
                    reward_metric,
                    progress_bar,
                    status_text
                )

            st.success("✅ Optimization completed!")

            # Display results
            st.markdown("---")
            st.subheader("📊 Results")

            # Metrics in columns
            metric_cols = st.columns(4)
            metrics = result['metrics']

            with metric_cols[0]:
                st.metric("IoU", f"{metrics['iou']:.4f}")
            with metric_cols[1]:
                st.metric("Dice", f"{metrics['dice']:.4f}")
            with metric_cols[2]:
                st.metric("F1 Score", f"{metrics['f1']:.4f}")
            with metric_cols[3]:
                st.metric("SSIM", f"{metrics['ssim']:.4f}")

            # Best pipeline
            st.subheader("🔧 Optimal Pipeline")
            pipeline_cols = st.columns(3)

            pipeline = result['pipeline']
            with pipeline_cols[0]:
                st.info(f"**Preprocessing**\n\n{pipeline['preprocessing'][1]}")
            with pipeline_cols[1]:
                st.success(f"**Edge Detection**\n\n{pipeline['processing'][1]}")
            with pipeline_cols[2]:
                st.warning(f"**Postprocessing**\n\n{pipeline['postprocessing'][1]}")

            # Visualization
            st.subheader("📈 Detailed Analysis")

            # Convert back to RGB for display
            original_display = original_np
            if len(original_display.shape) == 3:
                original_display = cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB)

            gt_display = gt_np
            if len(gt_display.shape) == 3:
                gt_display = cv2.cvtColor(gt_display, cv2.COLOR_BGR2GRAY)

            fig = create_comparison_plot(
                original_display if len(original_display.shape) == 2 else cv2.cvtColor(original_display,
                                                                                       cv2.COLOR_RGB2GRAY),
                gt_display,
                result['result'],
                metrics,
                pipeline,
                result['reward_history']
            )

            st.pyplot(fig)

            # Download button for result
            st.subheader("💾 Download Result")

            # Convert result to PIL Image for download
            result_pil = Image.fromarray(result['result'])

            # Save to bytes
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Download Edge Detection Result",
                data=byte_im,
                file_name="rl_edge_detection_result.png",
                mime="image/png",
                use_container_width=True
            )


if __name__ == "__main__":
    main()