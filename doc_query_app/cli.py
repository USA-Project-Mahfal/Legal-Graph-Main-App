#!/usr/bin/env python3
import argparse
from data_manager import DataManager


def main():
    parser = argparse.ArgumentParser(description='Document Query CLI')
    parser.add_argument('command', choices=['embeddings', 'graph', 'both'],
                        help='Command to execute (embeddings, graph, or both)')

    args = parser.parse_args()

    # Initialize data manager
    data_manager = DataManager()

    try:
        if args.command == 'embeddings':
            print("Regenerating embeddings...")
            data_manager.init_embeddings_and_pilot_model(force=True)
            print("✅ Embeddings regenerated successfully!")

        elif args.command == 'graph':
            print("Regenerating graph data...")
            data_manager.build_3D_graph(force=True)
            print("✅ Graph data regenerated successfully!")

        elif args.command == 'both':
            print("Regenerating embeddings and graph...")
            data_manager.init_embeddings_and_pilot_model(force=True)
            data_manager.build_3D_graph(force=True)
            print("✅ Both embeddings and graph regenerated successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
