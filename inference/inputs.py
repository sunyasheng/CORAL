"""Validate portable inputs before creating a model or allocating GPU memory."""
import copy
import json
import re
from pathlib import Path


def chain_name(index):
    name = ''
    while index:
        index, r = divmod(index - 1, 26)
        name = chr(65 + r) + name
    return name


def entities_by_chain(case):
    chains = {}
    for entity in case['sequences']:
        kind, info = next(iter(entity.items()))
        for _ in range(info.get('count', 1)):
            chains[chain_name(len(chains) + 1)] = (kind, info)
    return chains


def a3m_query(path, sequence, require_hits=True):
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('>'):
            records.append('')
        elif records:
            records[-1] += line
        else:
            raise ValueError(f'A3M sequence before header: {path}')
    if not records or len(records[0]) != len(sequence) or not all(a == b or a == 'X' for a,b in zip(sequence, records[0])):
        raise ValueError(f'A3M query must match the protein (input X may match a known residue): {path}')
    if require_hits and len(records) < 2:
        raise ValueError(f'True MSA requires more than a query-only A3M: {path}')
    for record in records:
        aligned = re.sub('[a-z.]', '', record)
        if len(aligned) != len(sequence) or not re.fullmatch('[A-Z-]+', aligned):
            raise ValueError(f'Invalid aligned A3M row: {path}')
    return len(records)


def load_input(path, mode, receptor=None):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError('Input must be a list containing exactly one case object')
    case = copy.deepcopy(data[0])
    if not isinstance(case.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', case['name']):
        raise ValueError('Input name must contain only letters, digits, underscores or hyphens')
    entities = case.get('sequences')
    if not isinstance(entities, list) or not entities:
        raise ValueError('sequences must be a nonempty list')
    proteins = []
    ligands = []
    msa_files = []
    for e in entities:
        if not isinstance(e, dict) or len(e) != 1 or next(iter(e)) not in ('proteinChain', 'ligand'):
            raise ValueError('Each entity must contain exactly one proteinChain or ligand')
        kind, info = next(iter(e.items()))
        if not isinstance(info, dict):
            raise ValueError('Entity value must be an object')
        info.setdefault('count', 1)
        if type(info['count']) is not int or info['count'] < 1:
            raise ValueError('Entity count must be a positive integer')
        if kind == 'proteinChain':
            seq = info.get('sequence')
            if not isinstance(seq, str) or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYX]+', seq):
                raise ValueError('Protein sequence must use uppercase amino-acid letters (or X)')
            proteins.append(info)
        else:
            if not isinstance(info.get('ligand'), str) or not info['ligand'].strip():
                raise ValueError('ligand must be CCD_<id> or a SMILES string')
            ligands.append(info)
        if mode == 'posebusters' and any(k in info for k in ('protofilament', 'layer', 'pf_id', 'layer_id')):
            raise ValueError('PoseBusters blind mode does not accept PF/layer declarations')
        if 'layer' in info and 'protofilament' not in info:
            raise ValueError('layer requires protofilament')
        for field in ('protofilament', 'layer'):
            if field in info:
                v = info[field]
                if not isinstance(v, list) or len(v) != info['count'] or any(type(x) is not int or x < 0 for x in v):
                    raise ValueError(f'{field} requires one nonnegative integer per copy')
                if field == 'protofilament' and any(x > 2 for x in v):
                    raise ValueError('protofilament must be 0, 1 or 2')
    if not proteins or not ligands:
        raise ValueError('Both proteinChain and ligand entities are required')
    if mode == 'posebusters':
        if receptor is not None or case.get('constraint') or case.get('constraints'):
            raise ValueError('PoseBusters blind mode takes no receptor or pocket/contact constraints')
        # This public protocol matches the single-ligand benchmark, including its PDB export.
        if sum(x['count'] for x in ligands) != 1 or len(entities_by_chain(case)) > 26:
            raise ValueError('PoseBusters mode requires one ligand copy and at most 26 chains')
        heteromer = len({x['sequence'] for x in proteins}) > 1
        msa_depths = []
        for protein in proteins:
            msa = protein.get('msa')
            if not isinstance(msa, dict) or not msa.get('precomputed_msa_dir'):
                raise ValueError('PoseBusters mode requires precomputed_msa_dir for every protein')
            mdir = Path(msa['precomputed_msa_dir'])
            if not mdir.is_absolute():
                mdir = path.parent / mdir
            mdir = mdir.resolve()
            msa['precomputed_msa_dir'] = str(mdir)
            if msa.get('pairing_db') not in ('uniref100', 'uniprot'):
                raise ValueError('MSA pairing_db must be uniref100 or uniprot')
            msa_depths.append(a3m_query(mdir / 'non_pairing.a3m', protein['sequence'], require_hits=False))
            if heteromer:
                a3m_query(mdir / 'pairing.a3m', protein['sequence'], require_hits=False)
            msa_files.extend(sorted(mdir.glob('*.a3m')))
        if max(msa_depths) <= 1:
            raise ValueError('At least one protein requires an MSA with aligned homologues')
    else:
        if receptor is None:
            raise ValueError('Fibril mode requires --receptor and its adjacent .refmap.json')
        all_pf_layers = []
        for protein in proteins:
            if 'protofilament' not in protein or 'layer' not in protein:
                raise ValueError('Fibril mode requires explicit protofilament and layer arrays')
            pairs=[(p,l) for p,l in zip(protein['protofilament'],protein['layer']) if p]
            all_pf_layers.extend(pairs)
            if len(pairs)!=len(set(pairs)):
                raise ValueError('Duplicate (protofilament, layer) for protein copies')
        if len(all_pf_layers) != len(set(all_pf_layers)):
            raise ValueError('Duplicate (protofilament, layer) across protein entities')
        if any('msa' in x for x in proteins):
            raise ValueError('Fibril mode disables MSA; remove msa fields to avoid silent omission')
        mapping = json.loads(Path(receptor).with_suffix('.refmap.json').read_text())
        if mapping.get('version') != 1 or any(mapping.get('sequence') != p['sequence'] for p in proteins):
            raise ValueError('Refmap version must be 1 and sequence must match every protein entity')
        copies = mapping.get('copies')
        if not isinstance(copies, list) or not copies:
            raise ValueError('Refmap requires a nonempty copies list')
        expected = {c for c,(k,_) in entities_by_chain(case).items() if k == 'proteinChain'}
        seen = set(); targets = set()
        for e in copies:
            if not isinstance(e, dict) or e.get('json_chain') in seen or e.get('json_chain') not in expected:
                raise ValueError('Refmap must name each protein chain exactly once')
            seen.add(e['json_chain'])
            if not isinstance(e.get('chain'), str) or not isinstance(e.get('positions'), dict) or not e['positions']:
                raise ValueError('Each mapped copy requires a chain and nonempty positions')
            for k,v in e['positions'].items():
                if not k.isdigit() or str(int(k)) != k or not 1 <= int(k) <= len(mapping['sequence']) or type(v) is not int:
                    raise ValueError('Invalid refmap sequence position or residue number')
                target = (e['chain'], v)
                if target in targets:
                    raise ValueError('Refmap assigns multiple positions/copies to one receptor residue')
                targets.add(target)
        if seen != expected:
            raise ValueError('Refmap does not cover all protein copies')
    validate_constraints(case)
    return [case], sorted(set(msa_files))


def validate_constraints(case):
    import math
    allowed = {'name', 'sequences', 'constraint'}
    if set(case) - allowed:
        raise ValueError(f'Unsupported task fields: {sorted(set(case)-allowed)}')
    constraint = case.get('constraint', {})
    if not isinstance(constraint, dict) or set(constraint) - {'contact', 'pockets'}:
        raise ValueError('constraint supports contact and pockets only')
    def endpoint(entity, copy, position=None):
        if type(entity) is not int or not 1 <= entity <= len(case['sequences']):
            raise ValueError('Constraint entity index out of range')
        kind,info = next(iter(case['sequences'][entity-1].items()))
        if type(copy) is not int or not 1 <= copy <= info['count']:
            raise ValueError('Constraint copy index out of range')
        if position is not None:
            length = len(info['sequence']) if kind == 'proteinChain' else 1
            if type(position) is not int or not 1 <= position <= length:
                raise ValueError('Constraint position out of range')
        return entity,copy
    def distance(obj):
        lo,hi = obj.get('min_distance',0),obj.get('max_distance')
        if any(type(v) not in (float,int) or not math.isfinite(v) for v in (lo,hi)) or not 0 <= lo <= hi or hi == 0:
            raise ValueError('Constraint distances must be finite: 0 <= min <= max, max > 0')
    contacts=constraint.get('contact',[]);pockets=constraint.get('pockets',[])
    if not isinstance(contacts,list) or not isinstance(pockets,list):
        raise ValueError('contact and pockets must be lists')
    for c in contacts:
        allowed={'entity1','copy1','position1','atom1','entity2','copy2','position2','atom2','min_distance','max_distance'}
        if not isinstance(c,dict) or set(c)-allowed:
            raise ValueError('Unsupported contact fields')
        pairs=[]
        for side in ('1','2'):
            pairs.append(endpoint(c['entity'+side],c['copy'+side],c['position'+side]))
            if 'atom'+side in c and (not isinstance(c['atom'+side],str) or not c['atom'+side]):
                raise ValueError('Contact atoms must be explicit atom names')
        if pairs[0]==pairs[1]:raise ValueError('Contact endpoints must be on different chains')
        distance(c)
    for pocket in pockets:
        if not isinstance(pocket,dict) or set(pocket)!={'binder_chain','contact_residues','max_distance'}:
            raise ValueError('Pocket requires binder_chain, contact_residues, max_distance')
        binder=pocket['binder_chain'];bp=endpoint(binder['entity'],binder['copy'])
        residues=pocket['contact_residues']
        if not isinstance(residues,list) or not residues:raise ValueError('Pocket requires contact residues')
        for residue in residues:
            rp=endpoint(residue['entity'],residue['copy'],residue['position'])
            if bp==rp:raise ValueError('Pocket residue and binder must be on different chains')
        distance(pocket)


EXPERIMENT_ENV = ('PF_ID_OFF','PF_ID_SPLIT','PF_SYM_MODE','PF_SYM_OFFSET',
    'PF_REPULSION_TARGET','PF_REPULSION_MAXSTEP','PF_REPULSION_TAPER',
    'INFER_GAMMA0','INFER_STEP_SCALE_ETA','INFER_NOISE_SCALE_LAMBDA',
    'LIG_INIT_MODE','LIG_INIT_SIGMA','DEBUG_TRAIN_NOISE','DEBUG_IS_LIGAND_ZERO')


def check_environment(environ):
    present=[k for k in EXPERIMENT_ENV if k in environ]
    if present:raise ValueError('Unset experimental inference overrides before running: '+', '.join(present))
