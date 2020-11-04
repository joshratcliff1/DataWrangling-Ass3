""" Module with functionalities for blocking based on a dictionary of records,
    where a blocking function must return a dictionary with block identifiers
    as keys and values being sets or lists of record identifiers in that block.
"""

# =============================================================================

def noBlocking(rec_dict):
  """A function which does no blocking but simply puts all records from the
     given dictionary into one block.

     Parameter Description:
       rec_dict : Dictionary that holds the record identifiers as keys and
                  corresponding list of record values
  """

  print("Run 'no' blocking:")
  print('  Number of records to be blocked: '+str(len(rec_dict)))
  print('')

  rec_id_list = list(rec_dict.keys())

  block_dict = {'all_rec':rec_id_list}

  return block_dict

# -----------------------------------------------------------------------------

def simpleBlocking(rec_dict, blk_attr_list):
  """Build the blocking index data structure (dictionary) to store blocking
     key values (BKV) as keys and the corresponding list of record identifiers.

     A blocking is implemented that simply concatenates attribute values.

     Parameter Description:
       rec_dict      : Dictionary that holds the record identifiers as keys
                       and corresponding list of record values
       blk_attr_list : List of blocking key attributes to use

     This method returns a dictionary with blocking key values as its keys and
     list of record identifiers as its values (one list for each block).

     Examples:
       If the blocking is based on 'postcode' then:
         block_dict = {'2000': [rec1_id, rec2_id, rec3_id, ...],
                       '2600': [rec4_id, rec5_id, ...],
                         ...
                      }
       while if the blocking is based on 'postcode' and 'gender' then:
         block_dict = {'2000f': [rec1_id, rec3_id, ...],
                       '2000m': [rec2_id, ...],
                       '2600f': [rec5_id, ...],
                       '2600m': [rec4_id, ...],
                        ...
                      }
  """

  block_dict = {}  # The dictionary with blocks to be generated and returned

  print('Run simple blocking:')
  print('  List of blocking key attributes: '+str(blk_attr_list))
  print('  Number of records to be blocked: '+str(len(rec_dict)))
  print('')

  for (rec_id, rec_values) in rec_dict.items():

    rec_bkv = ''  # Initialise the blocking key value for this record

    # Process selected blocking attributes
    #
    for attr in blk_attr_list:
      attr_val = rec_values[attr]
      rec_bkv += attr_val

    # Insert the blocking key value and record into blocking dictionary
    #
    if (rec_bkv in block_dict): # Block key value in block index

      # Only need to add the record
      #
      rec_id_list = block_dict[rec_bkv]
      rec_id_list.append(rec_id)

    else: # Block key value not in block index

      # Create a new block and add the record identifier
      #
      rec_id_list = [rec_id]

    block_dict[rec_bkv] = rec_id_list  # Store the new block

  return block_dict

# -----------------------------------------------------------------------------

def phoneticBlocking(rec_dict, blk_attr_list):
    """Build the blocking index data structure (dictionary) to store blocking
       key values (BKV) as keys and the corresponding list of record identifiers.

       A blocking is implemented that concatenates Soundex encoded values of
       attribute values.

       Parameter Description:
         rec_dict      : Dictionary that holds the record identifiers as keys
                         and corresponding list of record values
         blk_attr_list : List of blocking key attributes to use

       This method returns a dictionary with blocking key values as its keys and
       list of record identifiers as its values (one list for each block).
    """

    block_dict = {}  # The dictionary with blocks to be generated and returned

    print('Run phonetic blocking:')
    print('  List of blocking key attributes: '+str(blk_attr_list))
    print('  Number of records to be blocked: '+str(len(rec_dict)))
    print('')

    for (rec_id, rec_values) in rec_dict.items():

        rec_bkv = ''  # Initialise the blocking key value for this record
        attr_count = 0
        # Process selected blocking attributes
        #
        for attr in blk_attr_list:
            attr_count += 1
            attr_val = rec_values[attr]

            # *********** Implement Soundex function here *********
            if not attr_val:
                rec_bkv += 'Z999'
            else:
                for i in range(len(attr_val)):
                    if i == 0:
                        rec_bkv += attr_val[0].upper()
                    elif len(rec_bkv) >= (4*attr_count):
                        continue
                    elif len(rec_bkv) < (4*attr_count):
                        if attr_val[i] in ['b', 'f', 'p', 'v']:
                            rec_bkv += '1'
                        elif attr_val[i] in ['d', 't']:
                            rec_bkv += '3'
                        elif attr_val[i] in ['m', 'n']:
                            rec_bkv += '5'
                        elif attr_val[i] in ['c', 'g', 'j', 'k', 'q', 's', 'x', 'z']:
                            rec_bkv += '2'
                        elif attr_val[i] in ['l']:
                            rec_bkv += '4'
                        elif attr_val[i] in ['r']:
                            rec_bkv += '6'
                if len(rec_bkv) < (4*attr_count):
                    rec_bkv += '0' * ((4*attr_count) - len(rec_bkv))

        # Also think about how to handle empty attribute values

        # ************ End of your Soundex code *********************************

        # Insert the blocking key value and record into blocking dictionary
        #
        if (rec_bkv in block_dict): # Block key value in block index

          # Only need to add the record
          #
          rec_id_list = block_dict[rec_bkv]
          rec_id_list.append(rec_id)

        else: # Block key value not in block index

          # Create a new block and add the record identifier
          #
          rec_id_list = [rec_id]

        block_dict[rec_bkv] = rec_id_list  # Store the new block

    return block_dict

# -----------------------------------------------------------------------------

def slkBlocking(rec_dict, fam_name_attr_ind, giv_name_attr_ind, 
                dob_attr_ind, gender_attr_ind):
    """Build the blocking index data structure (dictionary) to store blocking
       key values (BKV) as keys and the corresponding list of record identifiers.

       This function should implement the statistical linkage key (SLK-581)
       blocking approach as used in real-world linkage applications:

       http://www.aihw.gov.au/WorkArea/DownloadAsset.aspx?id=60129551915

       A SLK-581 blocking key is the based on the concatenation of:
       - 3 letters of family name
       - 2 letters of given name
       - Date of birth
       - Sex

       Parameter Description:
         rec_dict          : Dictionary that holds the record identifiers as
                             keys and corresponding list of record values
         fam_name_attr_ind : The number (index) of the attribute that contains
                             family name (last name)
         giv_name_attr_ind : The number (index) of the attribute that contains
                             given name (first name)
         dob_attr_ind      : The number (index) of the attribute that contains
                             date of birth
         gender_attr_ind   : The number (index) of the attribute that contains
                             gender (sex)

       This method returns a dictionary with blocking key values as its keys and
       list of record identifiers as its values (one list for each block).
    """

    block_dict = {}  # The dictionary with blocks to be generated and returned

    print('Run SLK-581 blocking:')
    print('  Number of records to be blocked: '+str(len(rec_dict)))
    print('')

    for (rec_id, rec_values) in rec_dict.items():

        rec_bkv = ''  # Initialise the blocking key value for this record

        # TODO: Copied Sample Code
        # Get family name value
        #
        fam_name = rec_values[fam_name_attr_ind]

        if (fam_name == ''):
            rec_bkv += '999'
        else:
            fam_name = fam_name.replace('-', '')  # Remove non letter characters
            fam_name = fam_name.replace(",", '')
            fam_name = fam_name.replace('_', '')

            if (len(fam_name) >= 5):
                rec_bkv += (fam_name[1] + fam_name[2] + fam_name[4])
            elif (len(fam_name) >= 3):
                rec_bkv += (fam_name[1] + fam_name[2] + '2')
            elif (len(fam_name) >= 2):
                rec_bkv += (fam_name[1] + '22')

        # Get given name value
        #
        giv_name = rec_values[giv_name_attr_ind]

        if (giv_name == ''):
            rec_bkv += '99'
        else:
            giv_name = giv_name.replace('-', '')  # Remove non letter characters
            giv_name = giv_name.replace(",", '')
            giv_name = giv_name.replace('_', '')

            if (len(giv_name) >= 3):
                rec_bkv += (giv_name[1] + giv_name[2])
            elif (len(giv_name) >= 2):
                rec_bkv += (giv_name[1] + '2')

        # DoB structure we use: dd/mm/yyyy

        # Get date of birth
        #
        dob = rec_values[dob_attr_ind]

        dob_list = rec_values[dob_attr_ind].split('/')

        # Add some checks
        #
        if (len(dob_list[0]) < 2):
            dob_list[0] = '0' + dob_list[0]  # Add leading zero for days < 10
        if (len(dob_list[1]) < 2):
            dob_list[1] = '0' + dob_list[1]  # Add leading zero for months < 10

        dob = ''.join(dob_list)  # Create: ddmmyyyy

        assert len(dob) == 8, dob

        rec_bkv += dob

        # Get gender
        #
        gender = rec_values[gender_attr_ind].lower()

        if (gender == 'm'):
            rec_bkv += '1'
        elif (gender == 'f'):
            rec_bkv += '2'
        else:
            rec_bkv += '9'

        # TODO: End of copied Sample Code

        # Insert the blocking key value and record into blocking dictionary
        #
        if (rec_bkv in block_dict): # Block key value in block index

            # Only need to add the record
            #
            rec_id_list = block_dict[rec_bkv]
            rec_id_list.append(rec_id)

        else: # Block key value not in block index

            # Create a new block and add the record identifier
            #
            rec_id_list = [rec_id]

        block_dict[rec_bkv] = rec_id_list  # Store the new block

    return block_dict

# -----------------------------------------------------------------------------

# Extra task if you have time:
# - Implement canopy clustering based blocking as described in the lectures
#   and the Data Matching book

# -----------------------------------------------------------------------------

def printBlockStatistics(blockA_dict, blockB_dict):
  """Calculate and print some basic statistics about the generated blocks
  """

  print('Statistics of the generated blocks:')

  numA_blocks = len(blockA_dict)
  numB_blocks = len(blockB_dict)

  block_sizeA_list = []
  for rec_id_list in blockA_dict.values():  # Loop over all blocks
    block_sizeA_list.append(len(rec_id_list))

  block_sizeB_list = []
  for rec_id_list in blockB_dict.values():  # Loop over all blocks
    block_sizeB_list.append(len(rec_id_list))

  print('Dataset A number of blocks generated: %d' % (numA_blocks))
  print('    Minimum block size: %d' % (min(block_sizeA_list)))
  print('    Average block size: %.2f' % \
        (float(sum(block_sizeA_list)) / len(block_sizeA_list)))
  print('    Maximum block size: %d' % (max(block_sizeA_list)))
  print('')

  print('Dataset B number of blocks generated: %d' % (numB_blocks))
  print('    Minimum block size: %d' % (min(block_sizeB_list)))
  print('    Average block size: %.2f' % \
        (float(sum(block_sizeB_list)) / len(block_sizeB_list)))
  print('    Maximum block size: %d' % (max(block_sizeB_list)))
  print('')

# -----------------------------------------------------------------------------

# End of program.
