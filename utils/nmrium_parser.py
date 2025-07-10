# Standard library
from collections import Counter, defaultdict
import json
import math
from typing import Any, Callable, Dict, List, Tuple

# Third-party
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

class NMRIUMParser:
    """
    Class for parsing nmrium NMR data
    - Run NMRium at https://www.nmrium.org/predict
    - Draw molecule and predict spectrum
    - Export as > Save data as > save file locally .nmrium
    - Run this parser using the .nmrium file

    Parse file by running
    parser = NMRIUMParser("nmr.nmrium")

    Display the annotated RDKit molecule using
    parser.draw_annotated_molecule()

    Get the prompt for LLM benchmarking using
    parser.get_prompt()
    """

    nmr_experiments = {
        "1H": "PROTON_400MHz",
        "13C": "CARBON_400MHz",
        "COSY": "COSY_400MHz",
        "HSQC": "HSQC_400MHz",
        "HMBC": "HMBC_400MHz",
        # aliases
        "proton": "PROTON_400MHz",
        "carbon": "CARBON_400MHz",
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(self.file_path, "r") as fh:
            data = json.load(fh)["data"]

        self.mol = Chem.AddHs(Chem.MolFromMolBlock(data["molecules"][0]["molfile"], removeHs=False), addCoords=True)
        self.smiles: str = Chem.MolToSmiles(self.mol)
        self.formula: str = rdMolDescriptors.CalcMolFormula(self.mol)
        self.spectra: List[Dict[str, Any]] = data.get("spectra", [])

    # ------- Helper ----------
    def _find_spectrum(self, name):
        """Find the spectrum in the dataset"""
        try:
            return next(
                spec
                for spec in self.spectra
                if spec["info"]["name"] == self.nmr_experiments[name]
            )
        except StopIteration:
            raise ValueError(f"Spectrum “{name}” not found") from None

    def extract_1d_data(self,spec):
        """Extract 1D data (both proton and carbon NMR are extracted with this method"""
        all_peak_data = []

        # Each peak range is grouped
        for rng in spec["ranges"]["values"]:

            # Within each range, each signal has it's own entry
            for sig in rng.get("signals"):

                # Extract delta, multiplicity, number of atoms.
                # sig_atoms refers the atoms in the RDKit molecule that corespond to this signla
                sig_delta = sig["delta"]
                sig_mult = sig["multiplicity"]
                sig_nbAtoms = sig["nbAtoms"]
                sig_atoms = sig["atoms"]
                
                # For a given signal, collect all the j coupling constants
                js_vals = []
                for js in sig["js"]:
                    js_vals.append(js["coupling"])

                # Put all peak data into dictionary
                peak_dict = {
                    "delta": sig_delta,
                    "n_atoms":sig_nbAtoms,
                    "atoms":sig_atoms,
                    "js_vals":js_vals,
                    "sig_mult":sig_mult,
                }
                all_peak_data.append(peak_dict)

        # Sort in reverse order by chemical shift (NMR convention)
        all_peak_data.sort(key=lambda x: x["delta"], reverse=True)
        
        return all_peak_data
    
    def show_proton_nmr(self):
        """Create the print string for 1D proton NMR"""
    
        proton_nmr = self._find_spectrum("1H")
        all_peak_data = self.extract_1d_data(proton_nmr)
        
        # print peaks
        proton_nmr_str = "1H NMR: δ "
        for peak in all_peak_data:

            m_str = peak["sig_mult"]
            delta = round(peak["delta"], 2)
            n = peak["n_atoms"]
            
            if m_str == "s":
                proton_nmr_str += f"{delta} ({m_str}, {n}H), "
            else:
                j_str = ", ".join([str(round(j,2)) for j in peak["js_vals"]])
                proton_nmr_str += f"{delta} ({m_str}, J = {j_str} Hz, {n}H), "
                
        proton_nmr_str = proton_nmr_str.rstrip(", ") + "."
        return proton_nmr_str
    
    
    def show_carbon_nmr(self):
        """Create print string for 1D carbon NMR"""
        carbon_nmr = self._find_spectrum("13C")
        all_peak_data = self.extract_1d_data(carbon_nmr)
        
        # print peaks
        carbon_nmr_str = "13C NMR: δ "
        for peak in all_peak_data:
            delta = round(peak["delta"], 2)
            n = peak["n_atoms"]
            carbon_nmr_str += f"{delta} ({n}C, s), "
    
        carbon_nmr_str = carbon_nmr_str.rstrip(", ") + "."
        return carbon_nmr_str

    def _extract_2d_pairs(self, spec):
        """
        Extract 2D NMR data.
        - COSY
        - HSQC
        - HMBC
        Return list of deduplicated tuples describing cross peaks        
        """
        pairs = []
        for zone in spec.get("zones").get("values"):
            for sig in zone.get("signals"):
        
                x = round(sig["x"]["delta"], 2)
                y = round(sig["y"]["delta"], 2)
                
                # Discard the diagonal
                if x == y:
                    continue
                pairs.append((x,y))

        # Deduplicate (remove symmetry related peaks in COSY)
        seen = set()
        deduplicated = []
        for x, y in pairs:
            key = tuple(sorted((x, y)))
            if key not in seen:
                seen.add(key)
                deduplicated.append((x, y))
        return deduplicated

    def pairs_to_string(self, pairs):
        """Print tuple as string. Prints in form (x, y)"""
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        pair_string = ", ".join([f"({x}, {y})" for x,y in pairs])
        return pair_string
        
    def show_cosy_nmr(self):
        """Printer function for COSY NMR"""
        cosy_nmr = self._find_spectrum("COSY")
        pairs = self._extract_2d_pairs(cosy_nmr)
        cosy_string = f"COSY (δH, δH): {self.pairs_to_string(pairs)}."
        return cosy_string

    def show_hsqc_nmr(self):
        """Printer function for HSQC NMR"""
        hsqc_nmr = self._find_spectrum("HSQC")
        pairs = self._extract_2d_pairs(hsqc_nmr)
        hsqc_string = f"HSQC (δH, δC): {self.pairs_to_string(pairs)}."
        return hsqc_string

    def show_hmbc_nmr(self):
        """Printer function for HMBC NMR"""
        hmbc_nmr = self._find_spectrum("HMBC")
        pairs = self._extract_2d_pairs(hmbc_nmr)
        hmbc_string = f"HMBC (δH, δC): {self.pairs_to_string(pairs)}."
        return hmbc_string

    def show_formula(self):
        return f"Formula: {self.formula}"

    def get_prompt(self):
        return (
            "Write the SMILES string of the molecule consistent with this data.\n\n"
            f"{self.show_formula()}\n\n{self.show_proton_nmr()}\n\n{self.show_carbon_nmr()}\n\n"
            f"{self.show_cosy_nmr()}\n\n{self.show_hsqc_nmr()}\n\n{self.show_hmbc_nmr()}\n\n"
            "Only write the SMILES string. Do not write stereochemistry. Do not write any comments."
        )
    
    def draw_annotated_molecule(
            self,
            size: tuple[int, int] = (600, 400),
            out_png: str | None = None,
        ):
        """
        Render the molecule with ¹H and ¹³C chemical-shift labels taken from the
        NMRIUM file and highlight symmetry-related protons.
        """
    
        # ------------------------------------------------------------------ #
        # 1.  Build {NMRIUM_atom_index: ["H 7.26", "C 128.4", …]} dictionary
        # ------------------------------------------------------------------ #
        annotations: dict[int, list[str]] = defaultdict(list)
    
        def _collect(spec_name: str, nucleus: str):
            try:
                spec = self._find_spectrum(spec_name)
            except ValueError:
                return
            for peak in self.extract_1d_data(spec):
                label = f"{nucleus} {round(peak['delta'], 2)}"
                for idx in peak["atoms"]:
                    annotations[idx].append(label)

        _collect("1H", "H")
        _collect("13C", "C")
    
        if not annotations:
            raise RuntimeError("No atom annotations found in the NMRIUM data.")
    
        # ------------------------------------------------------------------ #
        # 2.  Prepare a copy of the molecule WITH explicit hydrogens
        # ------------------------------------------------------------------ #
        mol = Chem.Mol(self.mol)
    
        if not mol.GetNumConformers():               # ensure 2-D coordinates
            AllChem.Compute2DCoords(mol)
    
        n_atoms = mol.GetNumAtoms()
    
        # ------------------------------------------------------------------ #
        # 3.  Attach the notes (convert 1-based → 0-based)
        # ------------------------------------------------------------------ #
        for nmrium_idx, notes in annotations.items():
            rdkit_idx = nmrium_idx               # 1-based → 0-based
            if 0 <= rdkit_idx < n_atoms:
                mol.GetAtomWithIdx(rdkit_idx).SetProp("atomNote", "\n".join(notes))
            else:
                continue
        
        # ------------------------------------------------------------------ #
        # 5.  Draw the annotated molecule
        # ------------------------------------------------------------------ #
        w, h = size
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        png_bytes = drawer.GetDrawingText()
    
        if out_png:
            with open(out_png, "wb") as fh:
                fh.write(png_bytes)
            return
    
        from PIL import Image
        import io
        return Image.open(io.BytesIO(png_bytes))