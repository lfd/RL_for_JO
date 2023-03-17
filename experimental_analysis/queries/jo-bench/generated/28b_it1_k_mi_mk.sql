SELECT * FROM info_type AS it1, movie_keyword AS mk, movie_info AS mi, keyword AS k WHERE it1.info = 'countries' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;