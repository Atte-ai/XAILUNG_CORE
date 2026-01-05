import matplotlib.pyplot as plt
import os

def draw_xailung_architecture():
    # --- NEW: Safety check to create the visuals folder ---
    if not os.path.exists('visuals'):
        os.makedirs('visuals')
        print("Created 'visuals' directory.")

    # Set up the figure size for a dissertation page
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. Define Box Positions and Colors
    boxes = {
        "CT Volume (128x128x64)": (0.1, 0.75, "lightcyan"),
        "Histo Slide (224x224x3)": (0.1, 0.50, "honeydew"),
        "Clinical Data (Age/Smoke)": (0.1, 0.25, "mistyrose"),
        "3D CNN Branch": (0.35, 0.75, "skyblue"),
        "MobileNetV2 (Frozen)": (0.35, 0.50, "lightgreen"),
        "Metadata Dense Layer": (0.35, 0.25, "salmon"),
        "CONCATENATION": (0.60, 0.50, "plum"),
        "Fusion Dense (64)": (0.80, 0.50, "lightgrey"),
        "Sigmoid Output": (0.95, 0.50, "gold")
    }

    # 2. Draw the Boxes
    for text, (x, y, color) in boxes.items():
        ax.text(x, y, text, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.8', facecolor=color, edgecolor='black', alpha=0.9),
                fontsize=10, fontweight='bold')

    # 3. Draw the Connecting Arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    ax.annotate('', xy=(0.28, 0.75), xytext=(0.20, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.28, 0.50), xytext=(0.20, 0.50), arrowprops=arrow_props)
    ax.annotate('', xy=(0.28, 0.25), xytext=(0.20, 0.25), arrowprops=arrow_props)
    ax.annotate('', xy=(0.53, 0.52), xytext=(0.42, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.53, 0.50), xytext=(0.42, 0.50), arrowprops=arrow_props)
    ax.annotate('', xy=(0.53, 0.48), xytext=(0.42, 0.25), arrowprops=arrow_props)
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.67, 0.50), arrowprops=arrow_props)
    ax.annotate('', xy=(0.90, 0.50), xytext=(0.85, 0.50), arrowprops=arrow_props)

    ax.set_axis_off()
    plt.title("XAILUNG: Multimodal Late-Fusion Implementation Strategy", fontsize=14, fontweight='bold', pad=20)
    
    # Save the file
    plt.savefig('visuals/implementation_flowchart.png', dpi=300, bbox_inches='tight')
    print("Success! Your architecture diagram is saved in 'visuals/implementation_flowchart.png'")
    plt.show()

if __name__ == "__main__":
    draw_xailung_architecture()