from functools import reduce
from itertools import chain

import networkx as nx
import numpy as np
import torch
from PIL.Image import Image

from detector import crop_face
from configs import Config
from torch.nn import functional as F

cfg = Config()

def get_identities(frame, faces):
    # 同身份不同场景维护 => 由身份阈值确定场景可变性
    identities_dict = {}
    face = crop_face(frame)
    if face: faces.append(face)
    if len(faces) >= cfg.NUM_MAX_FRAMES:
        faces = list(chain.from_iterable(faces))
        identities = [torch.tensor(face['embedding'].tolist(), dtype=torch.float32, device=cfg.device) for face in faces]

        identities = torch.stack(identities, dim=0)
        identities = F.normalize(identities, dim=1)
        cosine_similarity = torch.mm(identities, identities.t())

        unknown_identities = np.array(range(cosine_similarity.shape[0]))

        cosine_similarity = torch.triu(cosine_similarity, diagonal=1)
        res = torch.where(cosine_similarity > cfg.identity_threshold)

        identity_indices = [r.detach().clone().tolist() for r in res]

        def merge_sets_networkx(sets):
            G = nx.Graph()
            for s in sets:
                nodes = list(s)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        G.add_edge(nodes[i], nodes[j])
            return list(nx.connected_components(G))

        identities_set = [set(i) for i in zip(*identity_indices)]
        known_identities = merge_sets_networkx(identities_set)
        union_known_identities = reduce(lambda a, b: a | b, known_identities, set())
        unknown_identities = set(unknown_identities) - union_known_identities
        identity_ids = 0

        for known in known_identities:
            # 保证时序一致
            perm = np.array(sorted(list(known)))
            identities_dict[identity_ids] = np.array(faces)[perm]
            identity_ids += 1

        for unknown in unknown_identities:
            identities_dict[identity_ids] = np.array([np.array(faces)[unknown]])

        # for identities in identities_dict.values():
        #     for ids in identities:
        #         Image.fromarray(ids['cropped']).show()
        #     print()
    return identities_dict