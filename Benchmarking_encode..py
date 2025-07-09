import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tabulate import tabulate
import os
import yaml
import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

### Obtenemos de cada modulo lo necesario para realizar el bnenchmarking
from models.transformer import BertImage
from models.bert import BertConfig
from utils.learning_rate import linear_warmup_cosine_lr_scheduler
from utils.plotting import plot_attention_positions_all_layers
from utils.logging import get_num_parameter, human_format
from tensorboardX import SummaryWriter

## Hiperparametros para el benchmark general
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_TRAIN_IMAGES =512
NUM_TEST_IMAGES = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_OUTPUT_DIR = "benchmark_results"


print(f"Cargando subconjunto de CIFAR-10 ({NUM_TRAIN_IMAGES} para entrenar, {NUM_TEST_IMAGES} para testeo)")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_subset = Subset(full_train_dataset, range(NUM_TRAIN_IMAGES))
test_subset = Subset(full_test_dataset, range(NUM_TEST_IMAGES))
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
CLASS_NAMES = full_train_dataset.classes

### Funcion para obtener matriz de confusión
def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, encoding_type):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Etiqueta Predicha', fontsize=12)
    ax.set_ylabel('Etiqueta Real', fontsize=12)
    ax.set_title(f'Matriz de Confusión - {encoding_type}', fontsize=16)
    plot_path = os.path.join(output_dir, f"confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {plot_path}")

### En esta funcion corremos el experimento como tal
def run_experiment(encoding_type: str):
    print("\n" + "="*60)
    print(f"--- Iniciando experimento para: {encoding_type} ---")
    print("="*60)
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, encoding_type)
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(logdir=output_dir)
    print(f"Resultados, logs y gráficos se guardarán en: {output_dir}")

    config = {
        "num_hidden_layers": 6, "num_attention_heads": 9, "hidden_size": 396,
        "intermediate_size": 512, "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1, "layer_norm_eps": 1e-12,
        "max_position_embeddings": 32, "position_encoding_size": -1,
        "pooling_use_resnet": False, "pooling_concatenate_size": 1,
        "share_position_encoding": False, "gaussian_init_sigma_std": 0.01,
        "gaussian_init_mu_std": 2.0
    }
    
    ## Aqui se setea cada eencoding a usar como tal
    if encoding_type == 'relativo_aprendido_contenido':
        config.update({"use_learned_2d_encoding": True, "relative_position_embedding": True, "use_gaussian_attention": False, "use_attention_data": True, "query_positional_score": True})
    elif encoding_type == 'relativo_aprendido_solo_posicion':
        config.update({"use_learned_2d_encoding": True, "relative_position_embedding": True, "use_gaussian_attention": False, "use_attention_data": False, "query_positional_score": True})
    elif encoding_type == 'cuadratico_gaussiano':
        config.update({"use_learned_2d_encoding": False, "relative_position_embedding": True, "use_gaussian_attention": True, "attention_isotropic_gaussian": True, "use_attention_data": False, "query_positional_score": False})
    elif encoding_type == 'absoluto':
        config.update({"use_learned_2d_encoding": False, "relative_position_embedding": False, "use_gaussian_attention": False, "use_attention_data": True, "query_positional_score": False})
    elif encoding_type == 'sin_encoding':
        config.update({"use_learned_2d_encoding": False, "relative_position_embedding": False, "use_gaussian_attention": False, "use_attention_data": True, "query_positional_score": False})

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    if encoding_type == 'absoluto':
        class BertImageAbsolute(BertImage):
            def __init__(self, config, num_classes):
                super().__init__(config, num_classes)
                self.abs_pos_embedding = nn.Parameter(torch.randn(1, 32*32, config['hidden_size']))
            def forward(self, batch_images, **kwargs):
                device = batch_images.device
                batch_features = batch_images.permute(0, 2, 3, 1)
                batch_features = self.features_upscale(batch_features)
                b, w, h, c = batch_features.shape
                batch_features = batch_features.view(b, w * h, c)
                batch_features = batch_features + self.abs_pos_embedding[:, :(w*h), :].to(device)
                _, all_representations = self.encoder(batch_features, attention_mask=self.attention_mask)
                representations = all_representations[0]
                cls_representation = representations.mean(dim=1)
                return self.classifier(cls_representation)
        model = BertImageAbsolute(config, num_classes=10)
    else:
        model = BertImage(config, num_classes=10)
    
    model.to(DEVICE)
    
    num_params, _ = get_num_parameter(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [{encoding_type}]")
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
            writer.add_scalar(f'loss/train', loss.item(), epoch * len(train_loader) + i)
        
        if encoding_type == 'cuadratico_gaussiano':
            print("Generando y guardando gráficos de atención")
            plot_attention_positions_all_layers(model, config['num_attention_heads'], writer, epoch)

    model.eval()
    all_labels = []
    all_preds = []
    total_test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * (np.array(all_labels) == np.array(all_preds)).sum() / len(all_labels)
    avg_test_loss = total_test_loss / len(test_loader)
    
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, output_dir, encoding_type)

    writer.add_scalar('loss/test', avg_test_loss, NUM_EPOCHS)
    writer.add_scalar('accuracy/test', accuracy, NUM_EPOCHS)
    print(f"Resultado para '{encoding_type}': Precisión = {accuracy:.2f}%, Pérdida de Test = {avg_test_loss:.4f}")
    writer.close()
    
    return {"accuracy": accuracy, "test_loss": avg_test_loss, "params": num_params}

def create_summary_plots(results, output_dir):
    print("\nGenerando gráficos de resumen")
    encoding_types = list(results.keys())
    accuracies = [r['accuracy'] for r in results.values()]
    durations = [r['duration'] for r in results.values()]
    test_losses = [r['test_loss'] for r in results.values()]
    params_in_millions = [r['params'] / 1e6 for r in results.values()]

    ## Gráfico 1: Precisión
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(encoding_types, accuracies, color='skyblue')
    ax.set_ylabel('Precisión (%)')
    ax.set_title('Precisión por Tipo de Encoding')
    ax.set_ylim(0, max(accuracies) * 1.2 if max(accuracies) > 0 else 10)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_benchmark_accuracy.png"))
    plt.close(fig)

    ## Gráfico 2: Pérdida de Test
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(encoding_types, test_losses, color='salmon')
    ax.set_ylabel('Pérdida (Loss)')
    ax.set_title('Pérdida de Test por Tipo de Encoding')
    for i, v in enumerate(test_losses):
        ax.text(i, v * 1.02, f"{v:.4f}", ha='center')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_benchmark_test_loss.png"))
    plt.close(fig)

    ## Gráfico 3: Tiempo de Ejecución
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(encoding_types, durations, color='lightcoral')
    ax.set_ylabel('Tiempo (segundos)')
    ax.set_title('Tiempo de Ejecución por Tipo de Encoding')
    for i, v in enumerate(durations):
        ax.text(i, v * 1.02, f"{v:.2f}s", ha='center')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_benchmark_time.png"))
    plt.close(fig)

    ## Gráfico 4: Complejidad vs. Rendimiento
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(params_in_millions, accuracies, color='purple', s=150, alpha=0.7, zorder=5)
    ax.set_xlabel('Número de Parámetros (en Millones)')
    ax.set_ylabel('Precisión (%)')
    ax.set_title('Complejidad del Modelo vs. Precisión')
    ax.grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(encoding_types):
        ax.annotate(txt, (params_in_millions[i], accuracies[i]), xytext=(5, -5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_benchmark_complexity_vs_accuracy.png"))
    plt.close(fig)
    
    print(f"4 gráficos de resumen separados guardados en la carpeta: {output_dir}")

if __name__ == "__main__":
    encodings_to_test = [
        'relativo_aprendido_contenido', 'relativo_aprendido_solo_posicion',
        'cuadratico_gaussiano', 'absoluto', 'sin_encoding'
    ]
    
    results = {}
    for encoding in encodings_to_test:
        start_time = time.time()
        metrics = run_experiment(encoding)
        end_time = time.time()
        duration = end_time - start_time
        results[encoding] = {**metrics, "duration": duration}

    ## Guardar resultados numéricos en JSON para revisar despues
    results_path = os.path.join(BASE_OUTPUT_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados numéricos detallados guardados en: {results_path}")

    # Guardar resumen en texto
    summary_path = os.path.join(BASE_OUTPUT_DIR, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("--- RESULTADOS FINALES DEL BENCHMARK (1 ÉPOCA) ---\n")
        table_data = [[key, f"{value['accuracy']:.2f}%", f"{value['test_loss']:.4f}", f"{human_format(value['params'])}", f"{value['duration']:.2f} s"] for key, value in results.items()]
        headers = ["Tipo de Encoding", "Precisión Final", "Pérdida de Test", "# Parámetros", "Tiempo de Ejecución"]
        f.write(tabulate(table_data, headers=headers))
    
    print("\n\n" + "="*80)
    print("--- RESULTADOS FINALES DEL BENCHMARK (1 ÉPOCA) ---")
    print("="*80)
    print(tabulate(table_data, headers=headers))
    
    create_summary_plots(results, BASE_OUTPUT_DIR)

    print("\nBenchmark completado.")
    print(f"Todos los resultados detallados se han guardado en la carpeta: '{BASE_OUTPUT_DIR}'")
    print("Para visualizar las métricas y gráficos de atención, abre una nueva terminal y ejecuta:")
    print(f"tensorboard --logdir {BASE_OUTPUT_DIR}")