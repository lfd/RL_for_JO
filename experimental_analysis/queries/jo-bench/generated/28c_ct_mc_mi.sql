SELECT * FROM company_type AS ct, movie_info AS mi, movie_companies AS mc WHERE mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;