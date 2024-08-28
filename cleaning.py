import csv
def load_node_ids(file_path, id_field):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        return set(row[id_field] for row in reader)

def validate_relationships(relationships_file, start_id_field, end_id_field, valid_start_ids, valid_end_ids):
    valid_rels = []
    invalid_rels = []
    with open(relationships_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start_id = row[start_id_field]
            end_id = row[end_id_field]
            if start_id in valid_start_ids and end_id in valid_end_ids:
                valid_rels.append(row)
            else:
                invalid_rels.append(row)
    return valid_rels, invalid_rels

def update_csv(file_path, valid_rels, fieldnames):
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rels)

def main():
    # File paths
    posts_file = 'posts.csv'
    users_file = 'users.csv'
    relationships_file = 'users_posts_rel.csv'
    updated_relationships_file = 'clean_users_posts_rel.csv'
    
    # Load node IDs
    post_node_ids = load_node_ids(posts_file, 'postId:ID(Post)')
    user_node_ids = load_node_ids(users_file, 'userId:ID(User)')
    
    # Validate relationships
    valid_rels, invalid_rels = validate_relationships(
        relationships_file,
        ':START_ID(User)',
        ':END_ID(Post)',
        user_node_ids,
        post_node_ids
    )
    
    # Print invalid relationships
    print("Invalid relationships found:")
    for rel in invalid_rels:
        print(rel)
    
    # Update the CSV with valid relationships
    update_csv(updated_relationships_file, valid_rels, [':START_ID(User)', ':END_ID(Post)'])
    
    print(f"Updated relationships saved to {updated_relationships_file}")

if __name__ == "__main__":
    main()
