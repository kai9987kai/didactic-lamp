import json
import matplotlib.pyplot as plt
import os
import sys

def create_visualizations(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        print("Run the C++ simulation first to generate this file.")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    ticks = data.get('ticks', [])
    if not ticks:
        print("Error: No tick data found in the JSON file.")
        sys.exit(1)
        
    gen_indices = [g['tick'] for g in ticks]
    
    # ── Create Figure ──
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Universe Simulation telemetry', fontsize=16)
    
    # 1. Population Dynamics (Herbivores vs Predators)
    ax1 = plt.subplot(2, 2, 1)
    herbivores = [g['herbivore_count'] for g in ticks]
    predators = [g['predator_count'] for g in ticks]
    alive = [h + p for h, p in zip(herbivores, predators)]
    
    ax1.plot(gen_indices, alive, 'k--', label='Total Alive', alpha=0.5)
    ax1.plot(gen_indices, herbivores, 'g-', label='Herbivores', linewidth=2)
    ax1.plot(gen_indices, predators, 'r-', label='Predators', linewidth=2)
    
    ax1.set_title('Trophic Population Dynamics')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Agent Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolutionary Metrics (Fitness & Novelty)
    ax2 = plt.subplot(2, 2, 2)
    mean_fitness = [g['mean_fitness'] for g in ticks]
    best_fitness = [g.get('max_fitness', g['mean_fitness']) for g in ticks] 
    
    ax2.plot(gen_indices, mean_fitness, 'b-', label='Mean Fitness', linewidth=2)
    ax2.plot(gen_indices, best_fitness, 'b--', label='Best Fitness', alpha=0.5)
    
    ax2.set_title('Evolutionary Progression')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Biome Distribution (Stacked Area)
    ax3 = plt.subplot(2, 2, 3)
    
    # Extract biome data
    biomes = ['Ocean', 'Tundra', 'Desert', 'Grassland', 'Forest', 'Jungle']
    biome_colors = ['#1E90FF', '#E0FFFF', '#F4A460', '#7CFC00', '#228B22', '#006400']
    
    biome_history = {b: [] for b in biomes}
    for g in ticks:
        b_dist = g.get('biome_distribution', {})
        for b in biomes:
            biome_history[b].append(b_dist.get(b, 0.0) * 100) # Convert to percentage
            
    # Stackplot
    ax3.stackplot(gen_indices, 
                  [biome_history[b] for b in biomes],
                  labels=biomes,
                  colors=biome_colors,
                  alpha=0.8)
                  
    ax3.set_title('World Biome Distribution')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('% of Landmass')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_ylim(0, 100)
    
    # 4. Ecosystem Health (Pheromones & Diversity)
    ax4 = plt.subplot(2, 2, 4)
    pheromones = [g['total_pheromone'] for g in ticks]
    diversity = [g['species_shannon'] for g in ticks]
    
    color1 = 'm'
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Total Pheromone', color=color1)
    ax4.plot(gen_indices, pheromones, color=color1, linewidth=2, label='Pheromone')
    ax4.tick_params(axis='y', labelcolor=color1)
    
    ax4_twin = ax4.twinx()
    color2 = 'c'
    ax4_twin.set_ylabel('Shannon Diversity', color=color2)
    ax4_twin.plot(gen_indices, diversity, color=color2, linewidth=2, label='Diversity', linestyle='--')
    ax4_twin.tick_params(axis='y', labelcolor=color2)
    
    ax4.set_title('Ecosystem Signaling & Diversity')
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = 'simulation_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Optionally show if not running headless
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display interactive plot (saved to file instead).")

if __name__ == "__main__":
    json_path = "simulation_summary.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    create_visualizations(json_path)
