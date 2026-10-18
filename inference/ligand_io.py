"""Export predicted coordinates with chemistry from the user's CCD/SMILES input."""
from collections import Counter
import pickle
from pathlib import Path
import numpy as np
from rdkit import Chem


class LigandTemplates:
    def __init__(self, data):
        self.data = Path(data)
        self.ccd = None
        self.cache = {}

    def get(self, ligand):
        if ligand not in self.cache:
            if ligand.startswith('CCD_'):
                code = ligand[4:]
                if '_' in code:
                    raise ValueError('SDF export supports one CCD component per ligand entity')
                if self.ccd is None:
                    with (self.data/'components.cif.rdkit_mol.pkl').open('rb') as f:
                        self.ccd = pickle.load(f)
                original = self.ccd.get(code)
                if original is None:
                    mol = self.from_cif(code)
                else:
                    mol = Chem.Mol(original)
                    for name,i in original.atom_map.items():
                        mol.GetAtomWithIdx(i).SetProp('coral_atom_name', name)
            else:
                mol = Chem.MolFromSmiles(ligand)
                if mol is None:
                    raise ValueError('Invalid ligand SMILES')
                mol = Chem.AddHs(mol)
                count = Counter()
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol().upper(); count[symbol] += 1
                    atom.SetProp('coral_atom_name', f'{symbol}{count[symbol]}')
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
            mol.RemoveAllConformers()
            self.cache[ligand] = mol
        return Chem.Mol(self.cache[ligand])

    def from_cif(self, code):
        # Newer CCD IDs may be present in components.cif but absent from the RDKit cache.
        import io
        from biotite.structure.io import pdbx
        lines=[]; active=False
        with (self.data/'components.cif').open() as f:
            for line in f:
                if line.startswith('data_'):
                    if active:break
                    active=line.strip() == 'data_'+code
                if active:lines.append(line)
        if not lines:raise ValueError(f'CCD component not found: {code}')
        block=pdbx.CIFFile.read(io.StringIO(''.join(lines)))[code]
        atoms=block['chem_comp_atom'];names=atoms['atom_id'].as_array(str).tolist()
        elements=atoms['type_symbol'].as_array(str).tolist()
        charges=atoms['charge'].as_array(int)
        mol=Chem.RWMol();lookup={n:i for i,n in enumerate(names)}
        for name,element,charge in zip(names,elements,charges):
            atom=Chem.Atom(element.capitalize());atom.SetFormalCharge(int(charge))
            atom.SetProp('coral_atom_name',name);mol.AddAtom(atom)
        if 'chem_comp_bond' in block:
            b=block['chem_comp_bond'];types={'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}
            for a,c,kind in zip(b['atom_id_1'].as_array(str),b['atom_id_2'].as_array(str),b['value_order'].as_array(str)):
                if kind not in types:raise ValueError(f'Unsupported CCD bond order {kind}')
                mol.AddBond(lookup[a],lookup[c],types[kind])
        mol=mol.GetMol();Chem.SanitizeMol(mol)
        xyz=np.column_stack([atoms[f'pdbx_model_Cartn_{axis}_ideal'].as_array(float) for axis in 'xyz'])
        if not np.isfinite(xyz).all():raise ValueError(f'CCD ideal coordinates missing: {code}')
        conf=Chem.Conformer(len(names));conf.Set3D(True)
        for i,point in enumerate(xyz):conf.SetAtomPosition(i,[float(x) for x in point])
        mol.AddConformer(conf);Chem.AssignStereochemistryFrom3D(mol)
        return mol

    def molecule(self, ligand, names, elements, xyz):
        mol = self.get(ligand)
        lookup = {a.GetProp('coral_atom_name'):a.GetIdx() for a in mol.GetAtoms()}
        if len(names) != len(set(names)) or set(names) != set(lookup):
            raise ValueError('Predicted ligand atom names do not match input chemistry')
        mol = Chem.RenumberAtoms(mol, [lookup[n] for n in names])
        if [a.GetSymbol().upper() for a in mol.GetAtoms()] != [str(e).upper() for e in elements]:
            raise ValueError('Predicted ligand elements do not match input chemistry')
        xyz = np.asarray(xyz)
        if xyz.shape != (len(names),3) or not np.isfinite(xyz).all():
            raise ValueError('Invalid ligand coordinates')
        conf = Chem.Conformer(len(names)); conf.Set3D(True)
        for i,point in enumerate(xyz):
            conf.SetAtomPosition(i, [float(x) for x in point])
        mol.AddConformer(conf)
        mol.SetProp('CORAL_ATOM_NAMES', ' '.join(names))
        mol.SetProp('CORAL_TOPOLOGY_SOURCE', 'input CCD/SMILES; predicted coordinates, no relaxation')
        return mol

    def write(self, path, atoms, case):
        from inputs import entities_by_chain
        writer = Chem.SDWriter(str(path))
        try:
            for chain,(kind,info) in entities_by_chain(case).items():
                if kind != 'ligand':continue
                part = atoms[atoms.chain_id == chain]
                mol = self.molecule(info['ligand'], list(part.atom_name), list(part.element), part.coord)
                mol.SetProp('_Name', f"{case['name']} chain {chain}")
                writer.write(mol)
        finally:
            writer.close()
