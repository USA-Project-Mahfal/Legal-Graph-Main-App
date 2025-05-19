# import argparse
# from data_manager import DataManager  # Import the class instead of singleton


# def main():
#     parser = argparse.ArgumentParser(description='Document Query CLI')
#     parser.add_argument('command', choices=['embeddings', 'graph', 'both'],
#                         help='Command to execute (embeddings, graph, or both)')
#     parser.add_argument('--lazy', action='store_true',
#                         help='Initialize DataManager lazily (skip initial embeddings and graph)')

#     args = parser.parse_args()

#     # Create instance without triggering full init
#     data_manager = DataManager(lazy_init=args.lazy)

#     try:
#         if args.command == 'embeddings':
#             print("Regenerating embeddings...")
#             data_manager.init_embeddings_and_pilot_model(force=True)
#             print("✅ Embeddings regenerated successfully!")

#         elif args.command == 'graph':
#             print("Regenerating graph data...")
#             data_manager.build_3D_graph(force=True)
#             print("✅ Graph data regenerated successfully!")

#         elif args.command == 'both':
#             print("Regenerating embeddings and graph...")
#             data_manager.init_embeddings_and_pilot_model(force=True)
#             data_manager.build_3D_graph(force=True)
#             print("✅ Both embeddings and graph regenerated successfully!")

#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         exit(1)


# if __name__ == "__main__":
#     main()
