from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures
import re
import ipdb
# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    # ipdb.set_trace()
    def add_emission_features(self, sequence, pos, y, features):
        #ipdb.set_trace()
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                    
        # Extended features below
        if 1:
            # First letter capital
            first_cap = word[0].isupper()
            # Generate feature name
            if first_cap and pos > 0:
                #feat_name = "firstcap:%s::%s" % (word[0], y_name) # includes first letter in feat_name
                feat_name = "firstcap::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)
            ##Generate feature name
            #feat_name = "firstcap:%s::%s" % (str(first_cap), y_name) # includes True or False in feat_name
            #feat_id = self.add_feature(feat_name)
            ## Append feature
            #if feat_id != -1:
            #    features.append(feat_id)
        if 1:
            # All capital
            all_caps = word.upper() == word
            # Generate feature name
            if all_caps:
                #feat_name = "allcaps:%s::%s" % (word, y_name)
                feat_name = "allcaps::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)
                
        if 1:
            # ed ending #MAD
            ed_ending = 'ed' == word[-2:]
            if ed_ending:
                # Generate feature name
                #feat_name = "ed_ending:%s::%s" % (word, y_name)
                feat_name = "ed_ending::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)
         
        if 1:
            # Prefixes
            max_prefix = 3
            for i in range(max_prefix):
                if len(word) > i+1:
                    prefix = word[:(i+1)]
                    # Generate feature name.
                    feat_name = "prefix:%s::%s" % (prefix, y_name)
                    # Get feature ID from name.
                    feat_id = self.add_feature(feat_name)
                    # Append feature.
                    if feat_id != -1:
                        features.append(feat_id)
        
        if 1:
            # No vowels
            all_cons = sum([i.lower() in ('a','e','i','o','u') for i in word]) == 0
            if all_cons:
                # Generate feature name
                #feat_name = "all_cons:%s::%s" % (word, y_name)
                feat_name = "all_cons::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)


        if 1:
            # All Vowels
            all_vowels = sum([i.lower() in ('a','e','i','o','u') for i in word]) == len(word)
            if all_vowels:
                # Generate feature name
                # feat_name = "all_vowels:%s::%s" % (str(all_vowels), y_name)
                #feat_name = "all_vowels:%s::%s" % (word, y_name)
                feat_name = "all_vowels::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)
        
        if 1:
            # Abbreviation
            pattern = re.compile("r\b[A-Z]{2,}\b")
            is_abbreviation = pattern.match(word) 
            if is_abbreviation:
                # Generate feature name
                #feat_name = "abbrev:%s::%s" % (word, y_name)
                feat_name = "abbrev::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)

        if 1:
            # Special chars
            special_chars = any(not c.isalnum() for c in word)
            if special_chars:
                # Generate feature name
                #feat_name = "special:%s::%s" % (word, y_name)
                feat_name = "special::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)

        if 1:
            # Numeric
            numeric = word.isnumeric()
            if numeric:
                # Generate feature name
                #feat_name = "numeric:%s::%s" % (word, y_name)
                feat_name = "numeric::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature
                if feat_id != -1:
                    features.append(feat_id)

        if 1:
            # Period
            if re.search(r".", word):
                #feat_name = "period:%s::%s" % (word, y_name)
                feat_name = "period::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        if 1:
            # Hyphen
            if re.search("-", word):
                #feat_name = "hyphen:%s::%s" % (word, y_name)
                feat_name = "hyphen::%s" % y_name
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)


        

        return features

class ExtendedUnicodeFeatures(UnicodeFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = y

        # Get word name from ID.
        x_name = x

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        feat_name = str(feat_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if str.istitle(word):
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.isdigit(word):
            # Generate feature name.
            feat_name = "number::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.find(word, "-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features
