import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from utils.brics_rec import FindBRICSBonds
from utils.recap_rec import FindRECAPBonds, FindBRBonds
from utils.junc_dec import FindJuncBonds, get_non_ring_edges_and_atoms
from utils.connect import ConnectBonds
from torch_geometric.data import Data, Batch

def chemfrag_bondidx_2(data, edge_mask , dec):
    bond_idx_list = []
    if 'raug' in dec:
        chem_bonds_list = FindRECAPBonds(data.mol)
    if 'baug' in dec:
        chem_bonds_list = FindBRICSBonds(data.mol)
    if 'braug' in dec:
        chem_bonds_list = FindBRBonds(data.mol)
    edges_list = data.edge_index.T.tolist()
    for bond in chem_bonds_list:
        try:
            bond_idx = edges_list.index(list(bond))
            if edge_mask[bond_idx]:
                bond_idx_list.append(bond_idx)
        except ValueError:
            # bond may not in edges list(for example RECAP)
            # print(f"Bond {bond} not found in edges_list.")
            continue

def chemfrag_bondidx(data, edge_mask , dec):
    bond_idx_list = []
    if 'recap' in dec:
        chem_bonds_list = FindRECAPBonds(data.mol)
    if 'brics' in dec:
        chem_bonds_list = FindBRICSBonds(data.mol)
    if dec == 'braug':
        chem_bonds_list = FindBRBonds(data.mol)
    edges_list = data.edge_index.T.tolist()
    for bond in chem_bonds_list:
        try:
            bond_idx = edges_list.index(list(bond))
            if edge_mask[bond_idx]:
                bond_idx_list.append(bond_idx)
        except ValueError:
            # bond may not in edges list(for example RECAP)
            # print(f"Bond {bond} not found in edges_list.")
            continue
    
    return bond_idx_list

def frag_aug(data, rotate_idx):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx) < 2:
        return data, data
    if len(rotate_idx) > 1:
        K = np.random.randint(1, len(rotate_idx))
    remove_edges_idx = np.random.choice(rotate_idx, K, replace=False)
    edge_list = augmented_data.edge_index.T.numpy()
    remove_edges = copy.deepcopy(augmented_data.edge_index.T.numpy())[remove_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(edge_list, remove_edges_idx, axis=0)
    iedge_list = edge_list_after_removal.tolist()
    mask_edges = np.zeros( len(iedge_list), dtype=bool)
    rotate_idx = []
    to_rotate = []
    for i in range(len(iedge_list)):
        if G2.has_edge(*iedge_list[i]):
            G2.remove_edge(*iedge_list[i])
            n_u = iedge_list[i][0]
            n_v = iedge_list[i][1]
            connected_components = nx.connected_components(G2)
            n_u_component_size = None
            n_u_component_index = None
            n_v_component_size = None
            n_v_component_index = None
            
            for index, component in enumerate(connected_components):
                if n_u in component:
                    n_u_component_size = len(component)
                    n_u_component_index = index
                elif n_v in component:
                    n_v_component_size = len(component)
                    n_v_component_index = index
                
                if n_u_component_size is not None and n_v_component_size is not None:
                    break
            if n_u_component_size is None or n_v_component_size is None:
                continue
            if n_u_component_size > n_v_component_size:
                rotate_index = iedge_list.index([n_u, n_v])
                rotate_idx.append(rotate_index)
                mask_edges[rotate_index] = True
                l = list(list(nx.connected_components(G2))[n_v_component_index])
                to_rotate.append(l)
            else:
                rotate_index = iedge_list.index([n_v, n_u])
                rotate_idx.append(rotate_index)
                mask_edges[rotate_index] = True
                l = list(list(nx.connected_components(G2))[n_u_component_index])
                to_rotate.append(l)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range( len(iedge_list) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    augmented_data.edge_mask = torch.tensor(mask_edges)
    augmented_data.mask_rotate = mask_rotate
    edge_index = augmented_data.edge_index
    edge_attr = augmented_data.edge_attr

    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    mask[remove_edges_idx] = False

    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask]
    augmented_data.edge_index = new_edge_index
    augmented_data.edge_attr = new_edge_attr
    data1 = data    
    data2 = augmented_data  
    return data, augmented_data


def frag_aug_conn(data, rotate_idx):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx) < 2:
        return data, data
    if len(rotate_idx) > 1:
        K = np.random.randint(1, len(rotate_idx))
    remove_edges_idx = np.random.choice(rotate_idx, K, replace=False)
    edge_list = augmented_data.edge_index.T.numpy()
    remove_edges = copy.deepcopy(augmented_data.edge_index.T.numpy())[remove_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(edge_list, remove_edges_idx, axis=0)
    iedge_list = edge_list_after_removal.tolist()
    mask_edges = np.zeros( len(iedge_list), dtype=bool)
    rotate_idx = []
    to_rotate = []
    G3 = copy.deepcopy(G2)
    for i in range(len(iedge_list)):
        if G3.has_edge(*iedge_list[i]):
            G3.remove_edge(*iedge_list[i])
            n_u = iedge_list[i][0]
            n_v = iedge_list[i][1]
            connected_components = nx.connected_components(G3)
            n_u_component_size = None
            n_u_component_index = None
            n_v_component_size = None
            n_v_component_index = None
            
            for index, component in enumerate(connected_components):
                if n_u in component:
                    n_u_component_size = len(component)
                    n_u_component_index = index
                elif n_v in component:
                    n_v_component_size = len(component)
                    n_v_component_index = index
                
                if n_u_component_size is not None and n_v_component_size is not None:
                    break
            if n_u_component_size is None or n_v_component_size is None:
                continue
                    # large part append the node index in smaller part
            if n_u_component_size > n_v_component_size:
                rotate_index = iedge_list.index([n_u, n_v])
                rotate_idx.append(rotate_index)
                mask_edges[rotate_index] = True
                l = list(list(nx.connected_components(G3))[n_v_component_index])
                to_rotate.append(l)
                # mask_edges.append(rotate_index)
            else:
                rotate_index = iedge_list.index([n_v, n_u])
                rotate_idx.append(rotate_index)
                mask_edges[rotate_index] = True
                l = list(list(nx.connected_components(G3))[n_u_component_index])
                to_rotate.append(l)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range( len(iedge_list) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    augmented_data.edge_mask = torch.tensor(mask_edges)
    augmented_data.mask_rotate = mask_rotate
    edge_index = augmented_data.edge_index
    edge_attr = augmented_data.edge_attr

    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    mask[remove_edges_idx] = False

    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask]
    augmented_data.edge_index = new_edge_index
    augmented_data.edge_attr = new_edge_attr
    data1 = data 
    data2 = augmented_data
    return data, augmented_data

def combine_mask_rotate(data_batch):
    total_cols = data_batch.x.size(0)
    mask_rotate = data_batch.mask_rotate
    total_rows = sum(mask.shape[0] for mask in mask_rotate)
    full_mask_rotate = np.zeros((total_rows, total_cols), dtype=bool)
    current_row = 0
    current_col = 0
    for mask in data_batch.mask_rotate:
        rows, cols = mask.shape
        full_mask_rotate[current_row:current_row+rows, current_col:current_col+cols] = mask
        current_row += rows
        current_col += cols
    return full_mask_rotate

def fragmentation(data, rotate_idx, rotate_idx_sel, z=20):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx_sel) < 2:
        data_batch = Batch.from_data_list([data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return data_batch
    if len(rotate_idx_sel) > 1:
        K = np.random.randint(1, min(len(rotate_idx_sel),5))
    remove_edges_idx = np.random.choice(rotate_idx_sel, K, replace=False)
    rest_edges_idx = np.setdiff1d(rotate_idx, remove_edges_idx)
    iedge_list = augmented_data.edge_index.T.numpy()
    iedge_attr = augmented_data.edge_attr
    remove_edges = copy.deepcopy(iedge_list)[remove_edges_idx]
    rest_edges = copy.deepcopy(iedge_list)[rest_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(iedge_list, remove_edges_idx, axis=0)
    mask = torch.ones(iedge_attr.size(0), dtype=torch.bool)
    mask[remove_edges_idx] = False
    edge_attr_after_removal = iedge_attr[mask]
    iedge_list = edge_list_after_removal.tolist()
    mask_edges = np.zeros( len(iedge_list), dtype=bool)
    G3 = copy.deepcopy(G2)
    rotate_idx = []
    to_rotate = []
    for i in range(len(rest_edges)):
        G3 = copy.deepcopy(G2)
        G3.remove_edge(*rest_edges[i])
        n_u = rest_edges[i][0]
        n_v = rest_edges[i][1]
        connected_components = nx.connected_components(G3)
        n_u_component_size = None
        n_u_component_index = None
        n_v_component_size = None
        n_v_component_index = None
        
        for index, component in enumerate(connected_components):
            if n_u in component:
                n_u_component_size = len(component)
                n_u_component_index = index
            elif n_v in component:
                n_v_component_size = len(component)
                n_v_component_index = index
            
            if n_u_component_size is not None and n_v_component_size is not None:
                break
        if n_u_component_size is None or n_v_component_size is None:
            continue
                # large part append the node index in smaller part
        if n_u_component_size > n_v_component_size:
            rotate_index = iedge_list.index([n_u, n_v])
            rotate_idx.append(rotate_index)
            mask_edges[rotate_index] = True
            l = list(list(nx.connected_components(G3))[n_v_component_index])
            to_rotate.append(l)
            # mask_edges.append(rotate_index)
        else:
            rotate_index = iedge_list.index([n_v, n_u])
            rotate_idx.append(rotate_index)
            mask_edges[rotate_index] = True
            l = list(list(nx.connected_components(G3))[n_u_component_index])
            to_rotate.append(l)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G2.nodes())), dtype=bool)
    for i in range( np.sum(mask_edges) ):
        mask_rotate[i][np.asarray(to_rotate[i], dtype=int)] = True
        # print(mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)])
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    connected_components = list(nx.connected_components(G2))
    new_data_list = []
    original_rotate_indices = [rotate_idx[i] for i in sorted_indices]
    for component in connected_components:
        component_nodes = list(component)
        node_index_map = {node: idx for idx, node in enumerate(component_nodes)}
        component_edges = []
        for u, v in G2.edges(component):
            if u in component and v in component:
                component_edges.append([u,v])
                component_edges.append([v,u])
        new_edge_index = np.array([[node_index_map[u], node_index_map[v]] for u, v in component_edges]).T
        edge_indices = [iedge_list.index([u, v]) for u, v in component_edges]
        new_edge_attr = edge_attr_after_removal[edge_indices]
        new_edge_mask = mask_edges[edge_indices]
        new_edge_list = [iedge_list[i] for i in edge_indices]
        new_mask_rotate = []
        new_rotate_order = []
        for original_index in original_rotate_indices:
            u, v = iedge_list[original_index]
            if u in component_nodes and v in component_nodes:
                original_mask = mask_rotate[original_rotate_indices.index(original_index)]
                new_mask = np.zeros(len(component_nodes), dtype=bool)
                for original_node_index in np.where(original_mask)[0]:
                    if original_node_index in node_index_map:
                        new_node_index = node_index_map[original_node_index]
                        new_mask[new_node_index] = True
                new_rotate_order.append(new_edge_list.index([u,v]) )
                new_mask_rotate.append(new_mask)
        if len(new_mask_rotate)<1:
            continue
        else:
            sorted_indices = sorted(range(len(new_rotate_order)), key=lambda x: new_rotate_order[x])
            new_mask_rotate = np.array([new_mask_rotate[i] for i in sorted_indices], dtype=bool)
        new_mask_rotate = np.array(new_mask_rotate, dtype=bool)
        pos = [pos[component_nodes] for pos in data.pos]
        w = data.weights
        new_data = Data(
            x= data.x[component_nodes], 
            edge_index=torch.tensor(new_edge_index, dtype=torch.long), 
            edge_attr=new_edge_attr, 
            z= data.z[component_nodes], 
            pos = pos,
            weights = w,
            canonical_smi = data.canonical_smi,
            mol = data.mol,
            edge_mask=torch.tensor(new_edge_mask),
            mask_rotate=new_mask_rotate
        )
        
        if(len(new_data.z)>z):
            new_data_list.append(new_data)
    if len(new_data_list)>0:
        data_batch = Batch.from_data_list([i for i in new_data_list] + [data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return data_batch
    else:
        data_batch = Batch.from_data_list([data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return data_batch
    return new_data_list



    # data merge
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])

#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     x = torch.cat([data1.x, data2.x], dim=0)
#     edge_index2 = data2.edge_index + data1.x.size(0)
#     edge_index = torch.cat([data1.edge_index, edge_index2], dim=1)
#     edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
#     pos = [torch.cat([data1.pos[0], data2.pos[0]], dim=0)]
#     z =  torch.cat([data1.z, data1.z], dim=0)
#     edge_mask = torch.cat([data1.edge_mask, data2.edge_mask], dim=0)
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     merged_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z = z,canonical_smi=data1.canonical_smi, mol=data1.mol ,pos= pos, weights= data1.weights, edge_mask=edge_mask, mask_rotate = mask_rotate)
    return data, augmented_data